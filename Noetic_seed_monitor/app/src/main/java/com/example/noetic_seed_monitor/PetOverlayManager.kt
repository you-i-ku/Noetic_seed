package com.example.noetic_seed_monitor

import android.animation.ValueAnimator
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.view.animation.AccelerateDecelerateInterpolator
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.random.Random

class PetOverlayManager(private val service: IkuMonitorService) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val windowManager = service.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private val prefs = service.getSharedPreferences("iku_pet_overlay", Context.MODE_PRIVATE)
    private var overlayView: PetOverlayView? = null
    private var layoutParams: WindowManager.LayoutParams? = null
    private var driftJob: Job? = null
    private var stateJob: Job? = null
    private var moveAnimator: ValueAnimator? = null

    var isShowing: Boolean = false
        private set

    fun show(): Boolean {
        if (isShowing) return true
        if (!canDrawOverlays()) {
            openOverlaySettings()
            return false
        }

        val view = PetOverlayView(service).apply {
            onDragDelta = { dx, dy ->
                setFacing(if (dx >= 0f) PetFacing.Right else PetFacing.Left)
                setSprite(PetSpriteState.Drag)
                moveBy(dx, dy)
            }
            onTap = { triggerTap() }
            onDrop = { triggerDrop(); savePosition() }
        }
        val x = prefs.getInt(KEY_X, defaultX())
        val y = prefs.getInt(KEY_Y, defaultY())
        val params = WindowManager.LayoutParams(
            dp(144),
            dp(160),
            overlayWindowType(),
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            val clamped = clampPosition(x, y)
            this.x = clamped.first
            this.y = clamped.second
        }

        return try {
            windowManager.addView(view, params)
            overlayView = view
            layoutParams = params
            isShowing = true
            startStateCollection()
            startDrift()
            true
        } catch (_: Exception) {
            overlayView = null
            layoutParams = null
            openOverlaySettings()
            false
        }
    }

    fun hide() {
        if (!isShowing) return
        moveAnimator?.cancel()
        driftJob?.cancel()
        stateJob?.cancel()
        overlayView?.let { view ->
            try {
                windowManager.removeView(view)
            } catch (_: Exception) {
            }
        }
        overlayView = null
        layoutParams = null
        isShowing = false
    }

    fun dispose() {
        hide()
        scope.cancel()
    }

    fun onApprovalDecision(approved: Boolean) {
        overlayView?.triggerApprovalDecision(approved)
    }

    private fun startStateCollection() {
        stateJob?.cancel()
        stateJob = scope.launch {
            var lastLog: String? = null
            var lastHistorySize = 0
            var cycleUntil = 0L
            var settleUntil = 0L

            IkuMonitorService.state.collectLatest { state ->
                val now = SystemClock.uptimeMillis()
                val latestLog = state.logLines.lastOrNull()
                if (latestLog != null && latestLog != lastLog) {
                    lastLog = latestLog
                    if (latestLog.startsWith("--- cycle ")) {
                        cycleUntil = now + CYCLE_ACTIVE_MS
                        settleUntil = 0L
                        overlayView?.triggerCycleStart()
                    }
                }

                val historySize = state.eHistory.size
                if (lastHistorySize > 0 && historySize > lastHistorySize) {
                    cycleUntil = 0L
                    settleUntil = now + SETTLE_MS
                    overlayView?.triggerCycleSettled()
                }
                lastHistorySize = historySize

                overlayView?.setSprite(decideSprite(state, now, cycleUntil, settleUntil))
                overlayView?.setDimmed(state.paused || state.phase is AppPhase.Disconnected || state.phase is AppPhase.Reconnecting)
            }
        }
    }

    private fun decideSprite(state: IkuState, now: Long, cycleUntil: Long, settleUntil: Long): PetSpriteState {
        return when {
            state.approvalRequests.isNotEmpty() -> PetSpriteState.ApprovalWaiting
            state.paused -> PetSpriteState.Paused
            state.phase is AppPhase.Disconnected || state.phase is AppPhase.Reconnecting -> PetSpriteState.Dormant
            now < settleUntil -> PetSpriteState.Settling
            now < cycleUntil -> PetSpriteState.CycleActive
            state.pressure > 0.35f -> PetSpriteState.Tense
            state.energy < 20f -> PetSpriteState.Paused
            else -> PetSpriteState.Idle
        }
    }

    private fun startDrift() {
        driftJob?.cancel()
        driftJob = scope.launch {
            while (true) {
                delay(4200L)
                val view = overlayView ?: continue
                if (view.isDragging) continue
                val params = layoutParams ?: continue
                val dx = Random.nextInt(-56, 57)
                val dy = Random.nextInt(-38, 39)
                view.setFacing(if (dx >= 0) PetFacing.Right else PetFacing.Left)
                view.setSprite(PetSpriteState.Moving)
                animateTo(params.x + dx, params.y + dy) {
                    view.setSprite(PetSpriteState.Idle)
                    savePosition()
                }
            }
        }
    }

    private fun moveBy(dx: Float, dy: Float) {
        moveAnimator?.cancel()
        val params = layoutParams ?: return
        val target = clampPosition(params.x + dx.roundToInt(), params.y + dy.roundToInt())
        params.x = target.first
        params.y = target.second
        updateViewLayout()
    }

    private fun animateTo(rawX: Int, rawY: Int, onEnd: (() -> Unit)? = null) {
        val params = layoutParams ?: return
        val target = clampPosition(rawX, rawY)
        val startX = params.x
        val startY = params.y
        moveAnimator?.cancel()
        moveAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 1800L
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener { animator ->
                val f = animator.animatedValue as Float
                params.x = (startX + (target.first - startX) * f).roundToInt()
                params.y = (startY + (target.second - startY) * f).roundToInt()
                updateViewLayout()
                if (f >= 1f) onEnd?.invoke()
            }
            start()
        }
    }

    private fun updateViewLayout() {
        val view = overlayView ?: return
        val params = layoutParams ?: return
        try {
            windowManager.updateViewLayout(view, params)
        } catch (_: Exception) {
        }
    }

    private fun savePosition() {
        val params = layoutParams ?: return
        prefs.edit().putInt(KEY_X, params.x).putInt(KEY_Y, params.y).apply()
    }

    private fun clampPosition(x: Int, y: Int): Pair<Int, Int> {
        val metrics = service.resources.displayMetrics
        val width = layoutParams?.width ?: dp(144)
        val height = layoutParams?.height ?: dp(160)
        val maxX = max(0, metrics.widthPixels - width)
        val maxY = max(0, metrics.heightPixels - height - dp(16))
        return x.coerceIn(0, maxX) to y.coerceIn(dp(16), maxY)
    }

    private fun canDrawOverlays(): Boolean {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.M || Settings.canDrawOverlays(service)
    }

    private fun openOverlaySettings() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return
        val intent = Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            Uri.parse("package:${service.packageName}"),
        ).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
        try {
            service.startActivity(intent)
        } catch (_: Exception) {
        }
    }

    private fun overlayWindowType(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }
    }

    private fun defaultX(): Int = service.resources.displayMetrics.widthPixels - dp(168)

    private fun defaultY(): Int = service.resources.displayMetrics.heightPixels / 3

    private fun dp(value: Int): Int = (value * service.resources.displayMetrics.density).roundToInt()

    private companion object {
        const val KEY_X = "pet_x"
        const val KEY_Y = "pet_y"
        const val CYCLE_ACTIVE_MS = 90_000L
        const val SETTLE_MS = 3_000L
    }
}

private enum class PetFacing { Left, Right }

private enum class PetSpriteState(val row: Int, val frames: Int, val frameMs: Long, val loop: Boolean = true) {
    Idle(row = 0, frames = 6, frameMs = 140L),
    Moving(row = 7, frames = 6, frameMs = 95L),
    Drag(row = 7, frames = 6, frameMs = 95L),
    Tap(row = 3, frames = 4, frameMs = 115L, loop = false),
    Drop(row = 4, frames = 5, frameMs = 105L, loop = false),
    CycleActive(row = 8, frames = 6, frameMs = 90L),
    Settling(row = 4, frames = 5, frameMs = 120L, loop = false),
    ApprovalWaiting(row = 6, frames = 6, frameMs = 130L),
    ApprovalYes(row = 3, frames = 4, frameMs = 110L, loop = false),
    ApprovalNo(row = 5, frames = 8, frameMs = 100L, loop = false),
    Paused(row = 6, frames = 6, frameMs = 220L),
    Dormant(row = 5, frames = 8, frameMs = 180L),
    Tense(row = 8, frames = 6, frameMs = 80L),
}

private class PetOverlayView(context: Context) : View(context) {
    var onDragDelta: ((Float, Float) -> Unit)? = null
    var onTap: (() -> Unit)? = null
    var onDrop: (() -> Unit)? = null
    var isDragging: Boolean = false
        private set

    private val atlas: Bitmap = BitmapFactory.decodeResource(resources, R.drawable.iku_pet_spritesheet)
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    private val source = Rect()
    private val dest = RectF()
    private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop
    private var spriteState = PetSpriteState.Idle
    private var fallbackState: PetSpriteState? = null
    private var stateStartedAt = SystemClock.uptimeMillis()
    private var facing = PetFacing.Right
    private var dimmed = false
    private var downRawX = 0f
    private var downRawY = 0f
    private var lastRawX = 0f
    private var lastRawY = 0f

    fun setSprite(nextState: PetSpriteState) {
        if (spriteState == nextState) return
        if (!spriteState.loop && SystemClock.uptimeMillis() - stateStartedAt < spriteState.frames * spriteState.frameMs) {
            fallbackState = nextState
            return
        }
        spriteState = nextState
        fallbackState = null
        stateStartedAt = SystemClock.uptimeMillis()
        invalidate()
    }

    fun setFacing(nextFacing: PetFacing) {
        facing = nextFacing
    }

    fun setDimmed(nextDimmed: Boolean) {
        dimmed = nextDimmed
        invalidate()
    }

    fun triggerTap() {
        spriteState = PetSpriteState.Tap
        fallbackState = PetSpriteState.Idle
        stateStartedAt = SystemClock.uptimeMillis()
        invalidate()
    }

    fun triggerDrop() {
        spriteState = PetSpriteState.Drop
        fallbackState = PetSpriteState.Idle
        stateStartedAt = SystemClock.uptimeMillis()
        invalidate()
    }

    fun triggerCycleStart() {
        spriteState = PetSpriteState.CycleActive
        fallbackState = null
        stateStartedAt = SystemClock.uptimeMillis()
        invalidate()
    }

    fun triggerCycleSettled() {
        spriteState = PetSpriteState.Settling
        fallbackState = PetSpriteState.Idle
        stateStartedAt = SystemClock.uptimeMillis()
        invalidate()
    }

    fun triggerApprovalDecision(approved: Boolean) {
        spriteState = if (approved) PetSpriteState.ApprovalYes else PetSpriteState.ApprovalNo
        fallbackState = PetSpriteState.Idle
        stateStartedAt = SystemClock.uptimeMillis()
        invalidate()
    }

    override fun performClick(): Boolean {
        super.performClick()
        onTap?.invoke()
        return true
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                isDragging = false
                downRawX = event.rawX
                downRawY = event.rawY
                lastRawX = event.rawX
                lastRawY = event.rawY
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val totalDx = event.rawX - downRawX
                val totalDy = event.rawY - downRawY
                if (!isDragging && max(abs(totalDx), abs(totalDy)) > touchSlop) {
                    isDragging = true
                }
                if (isDragging) {
                    onDragDelta?.invoke(event.rawX - lastRawX, event.rawY - lastRawY)
                }
                lastRawX = event.rawX
                lastRawY = event.rawY
                return true
            }
            MotionEvent.ACTION_UP -> {
                if (isDragging) {
                    onDrop?.invoke()
                } else {
                    performClick()
                }
                isDragging = false
                return true
            }
            MotionEvent.ACTION_CANCEL -> {
                isDragging = false
                return true
            }
        }
        return true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val elapsed = SystemClock.uptimeMillis() - stateStartedAt
        if (!spriteState.loop && elapsed >= spriteState.frames * spriteState.frameMs) {
            spriteState = fallbackState ?: PetSpriteState.Idle
            fallbackState = null
            stateStartedAt = SystemClock.uptimeMillis()
        }
        val frame = if (spriteState.loop) {
            ((SystemClock.uptimeMillis() - stateStartedAt) / spriteState.frameMs % spriteState.frames).toInt()
        } else {
            ((SystemClock.uptimeMillis() - stateStartedAt) / spriteState.frameMs).toInt().coerceIn(0, spriteState.frames - 1)
        }

        source.set(
            frame * CELL_WIDTH,
            spriteState.row * CELL_HEIGHT,
            frame * CELL_WIDTH + CELL_WIDTH,
            spriteState.row * CELL_HEIGHT + CELL_HEIGHT,
        )
        val scale = minOf(width / CELL_WIDTH.toFloat(), height / CELL_HEIGHT.toFloat())
        val drawW = CELL_WIDTH * scale
        val drawH = CELL_HEIGHT * scale
        dest.set((width - drawW) / 2f, (height - drawH) / 2f, (width + drawW) / 2f, (height + drawH) / 2f)

        paint.colorFilter = if (dimmed) DIM_FILTER else null
        if (facing == PetFacing.Left) {
            canvas.save()
            canvas.scale(-1f, 1f, width / 2f, height / 2f)
            canvas.drawBitmap(atlas, source, dest, paint)
            canvas.restore()
        } else {
            canvas.drawBitmap(atlas, source, dest, paint)
        }
        if (isAttachedToWindow) postInvalidateOnAnimation()
    }

    private companion object {
        const val CELL_WIDTH = 192
        const val CELL_HEIGHT = 208
        val DIM_FILTER = ColorMatrixColorFilter(
            ColorMatrix().apply {
                setSaturation(0.55f)
                postConcat(ColorMatrix(floatArrayOf(
                    0.72f, 0f, 0f, 0f, 0f,
                    0f, 0.78f, 0f, 0f, 0f,
                    0f, 0f, 0.95f, 0f, 12f,
                    0f, 0f, 0f, 0.82f, 0f,
                )))
            }
        )
    }
}