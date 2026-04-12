package com.example.noetic_seed_monitor

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.PowerSettingsNew
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import android.content.Context
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch
import java.util.Date
import java.text.SimpleDateFormat

class MainActivity : ComponentActivity() {

    private val vm: IkuViewModel by
    viewModels()

    // 撮影中の一時ファイル（TakePicture は URI に直接書く）
    private var pendingCameraFile: java.io.File? = null
    private var pendingCameraUri: android.net.Uri? = null

    // Activity-level カメラlauncher（Compose外で登録）
    // TakePicture: フル解像度の JPEG を指定 URI に書き込む
    private val cameraLauncher = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.TakePicture()
    ) { success ->
        val file = pendingCameraFile
        pendingCameraFile = null
        pendingCameraUri = null
        if (!success || file == null || !file.exists() || file.length() == 0L) {
            vm.finishCameraCapture(null, null)
            file?.delete()
            return@registerForActivityResult
        }
        try {
            val jpegBytes = file.readBytes()
            // 画像サイズ取得（デコードせず寸法だけ読む）
            val opts = android.graphics.BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            android.graphics.BitmapFactory.decodeFile(file.absolutePath, opts)
            // 長辺 1920px を超えるならリサイズ（LM Studio vision の既知制約）
            val maxSide = 1920
            val longest = maxOf(opts.outWidth, opts.outHeight)
            val finalBytes: ByteArray
            val finalW: Int
            val finalH: Int
            if (longest > maxSide) {
                val scale = maxSide.toFloat() / longest
                val newW = (opts.outWidth * scale).toInt()
                val newH = (opts.outHeight * scale).toInt()
                val sampleSize = run {
                    var s = 1
                    while (opts.outWidth / (s * 2) >= newW) s *= 2
                    s
                }
                val decodeOpts = android.graphics.BitmapFactory.Options().apply {
                    inSampleSize = sampleSize
                }
                val decoded = android.graphics.BitmapFactory.decodeFile(file.absolutePath, decodeOpts)
                val scaled = android.graphics.Bitmap.createScaledBitmap(decoded, newW, newH, true)
                decoded.recycle()
                val baos = java.io.ByteArrayOutputStream()
                scaled.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, baos)
                finalBytes = baos.toByteArray()
                finalW = scaled.width
                finalH = scaled.height
                scaled.recycle()
            } else {
                finalBytes = jpegBytes
                finalW = opts.outWidth
                finalH = opts.outHeight
            }
            val b64 = android.util.Base64.encodeToString(finalBytes, android.util.Base64.NO_WRAP)
            val now = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(java.util.Date())
            val meta = mutableMapOf<String, Any>(
                "captured_at" to now,
                "facing" to vm.pendingCameraFacing,
                "width" to finalW,
                "height" to finalH,
                "size_bytes" to finalBytes.size,
            )
            vm.finishCameraCapture(b64, meta)
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "camera read failed", e)
            vm.finishCameraCapture(null, null)
        } finally {
            file.delete()
        }
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchCameraWithTempFile()
        } else {
            vm.finishCameraCapture(null, null)
        }
    }

    private fun launchCameraStreamActivity(facing: String, frames: Int, intervalSec: Float) {
        try {
            // async モード: Activity がフレーム毎に直接 WebSocket 送信するためのコールバック
            CameraStreamBridge.sendMessage = { json -> vm.sendWsMessage(json) }
            // 旧バッチ版の onComplete も一応セット（Activity が null を渡して終了通知する）
            CameraStreamBridge.onComplete = { _ ->
                // async 版では使わない。Activity 側で stream_end を送ってくれる
                vm.finishCameraStream(null, null)
            }
            val intent = android.content.Intent(this, CameraStreamActivity::class.java).apply {
                putExtra(CameraStreamActivity.EXTRA_FACING, facing)
                putExtra(CameraStreamActivity.EXTRA_FRAMES, frames)
                putExtra(CameraStreamActivity.EXTRA_INTERVAL_SEC, intervalSec)
            }
            startActivity(intent)
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "launchCameraStreamActivity failed", e)
            CameraStreamBridge.sendMessage = null
            CameraStreamBridge.onComplete = null
            vm.finishCameraStream(null, null)
        }
    }

    private fun launchCameraWithTempFile() {
        try {
            val dir = java.io.File(cacheDir, "captures").apply { mkdirs() }
            val ts = java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(java.util.Date())
            val file = java.io.File(dir, "cap_$ts.jpg")
            val uri = androidx.core.content.FileProvider.getUriForFile(
                this, "${packageName}.fileprovider", file
            )
            pendingCameraFile = file
            pendingCameraUri = uri
            cameraLauncher.launch(uri)
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "launchCamera failed", e)
            vm.finishCameraCapture(null, null)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Foreground Service を早めに起動（バックグラウンド動作のため）
        IkuMonitorService.setContext(this)
        IkuMonitorService.startIfNeeded(this)

        // ナビゲーションバーを隠す（スワイプで一時表示）
        val controller = androidx.core.view.WindowCompat.getInsetsController(window, window.decorView)
        controller.hide(androidx.core.view.WindowInsetsCompat.Type.navigationBars())
        controller.systemBarsBehavior =
            androidx.core.view.WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE

        // ViewModelにカメラ起動トリガーを登録
        vm.setCameraTrigger {
            val hasCam = androidx.core.content.ContextCompat.checkSelfPermission(
                this, android.Manifest.permission.CAMERA
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
            if (hasCam) {
                launchCameraWithTempFile()
            } else {
                cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
            }
        }

        // ViewModelに camera_stream 起動トリガーを登録
        vm.setCameraStreamTrigger { facing, frames, intervalSec ->
            val hasCam = androidx.core.content.ContextCompat.checkSelfPermission(
                this, android.Manifest.permission.CAMERA
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
            if (hasCam) {
                launchCameraStreamActivity(facing, frames, intervalSec)
            } else {
                // 権限なしならコールバック null で通知
                vm.finishCameraStream(null, null)
            }
        }

        setContent {
            IkuApp(vm = vm)
        }
    }
}

fun entropyToColor(entropy: Float): Color {
    val r: Float; val g: Float; val b: Float
    when {
        entropy < 0.2f -> { r = 0.05f; g = 0.15f + entropy * 2f; b = 0.6f + entropy }
        entropy < 0.4f -> { r = 0.05f + (entropy - 0.2f) * 1.5f; g = 0.35f + (entropy - 0.2f); b = 0.5f - (entropy - 0.2f) * 2f }
        entropy < 0.6f -> { r = 0.35f + (entropy - 0.4f) * 2f; g = 0.55f - (entropy - 0.4f) * 1.5f; b = 0.1f }
        entropy < 0.8f -> { r = 0.75f + (entropy - 0.6f); g = 0.25f - (entropy - 0.6f); b = 0.05f }
        else -> { r = 0.9f; g = 0.1f - (entropy - 0.8f) * 0.5f; b = 0.05f + (entropy - 0.8f) * 0.3f }
    }
    return Color(r.coerceIn(0f, 1f), g.coerceIn(0f, 1f), b.coerceIn(0f, 1f))
}

// ===== Navigation =====
// Main/Dashboard/Terminal/Test は HorizontalPager の4ページ
// Settings は独立画面（Pager とは別経路、ドロワー経由でのみ到達）
enum class Screen { Main, Dashboard, Terminal, Test, Settings }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun IkuApp(vm: IkuViewModel = viewModel()) {
    val context = LocalContext.current
    LaunchedEffect(Unit) { vm.setContext(context) }

    // Android 13+ 通知権限リクエスト
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        val launcher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { _ -> }
        LaunchedEffect(Unit) {
            launcher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    // カメラ・マイク権限を起動時にプリリクエスト
    // （camera_stream / mic_record は Service context から Activity を起動するので、
    //  Activity 経由の requestPermission ができない。先に取っておく）
    val camPermLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ -> }
    val micPermLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ -> }
    LaunchedEffect(Unit) {
        val camGranted = androidx.core.content.ContextCompat.checkSelfPermission(
            context, Manifest.permission.CAMERA
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        if (!camGranted) {
            camPermLauncher.launch(Manifest.permission.CAMERA)
        }
        val micGranted = androidx.core.content.ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        if (!micGranted) {
            micPermLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    val state by vm.state.collectAsState()
    val phase = state.phase

    // Activity 生存中のみ保持。閉じたらリセット → 次回は ConnectScreen から。
    // これにより「ネット瞬断 → メイン画面維持」と「アプリ再起動 → ConnectScreen」を自然に区別。
    var userConfirmed by remember { mutableStateOf(false) }

    // プロファイル選択 → 起動前に LLM 設定を経由させるための中間状態
    var pendingProfileName by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(phase) {
        if (phase !is AppPhase.ProfileSelect) pendingProfileName = null
        // セッション TTL 超過で Disconnected に落ちた場合もリセット
        if (phase is AppPhase.Disconnected) userConfirmed = false
    }

    if (!userConfirmed) {
        // Connect ボタンを通るまで何も出さない
        ConnectScreen(
            context = context,
            onConnect = { url, token ->
                vm.connect(url, token)
                userConfirmed = true
            },
        )
    } else {
        when (phase) {
            is AppPhase.Disconnected -> {
                // セッション死亡 → ConnectScreen に戻す
                userConfirmed = false
            }
            is AppPhase.Reconnecting -> {
                // ネット瞬断（Activity 生存中）→ メイン画面 + バナー or Loading
                if (state.session != null) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        AppWithDrawer(state = state, vm = vm)
                        // 再接続中バナー
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .align(Alignment.TopCenter)
                                .padding(top = 36.dp, start = 16.dp, end = 16.dp)
                                .clip(RoundedCornerShape(8.dp))
                                .background(Color(0xFFEF5350).copy(alpha = 0.9f))
                                .padding(horizontal = 16.dp, vertical = 10.dp),
                        ) {
                            Text(
                                "再接続中...",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.fillMaxWidth(),
                                textAlign = TextAlign.Center,
                            )
                        }
                    }
                } else {
                    LoadingScreen("再接続中...")
                }
            }
            is AppPhase.WaitingForServer -> LoadingScreen("サーバ応答待ち...")
            is AppPhase.ProfileSelect -> {
                val pending = pendingProfileName
                if (pending != null) {
                    PreLaunchConfigScreen(
                        profileName = pending,
                        state = state,
                        onRequestProviders = { vm.requestLlmProviders() },
                        onCancel = { pendingProfileName = null },
                        onConfirm = { provider, model, apiKey, baseUrl ->
                            vm.setLlm(provider, model, apiKey, baseUrl)
                            vm.selectProfile(pending)
                        },
                    )
                } else {
                    ProfileSelectScreen(
                        profiles = phase.profiles,
                        onSelect = { name ->
                            pendingProfileName = name
                            vm.requestLlmProviders()
                        },
                    )
                }
            }
            is AppPhase.ProfileLoading -> LoadingScreen("${phase.name} 起動中...")
            is AppPhase.Running -> AppWithDrawer(state = state, vm = vm)
        }
    }
}

@Composable
fun LoadingScreen(message: String) {
    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0A0A1A)) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("noetic-seed", style = MaterialTheme.typography.headlineMedium, color = Color(0xFF4FC3F7))
            Spacer(modifier = Modifier.height(32.dp))
            CircularProgressIndicator(color = Color(0xFF4FC3F7))
            Spacer(modifier = Modifier.height(16.dp))
            Text(message, color = Color.Gray, fontSize = 13.sp)
        }
    }
}

// CameraCaptureDialog は IkuApp 内のLaunchedEffectに統合済み

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppWithDrawer(state: IkuState, vm: IkuViewModel) {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    var currentScreen by remember { mutableStateOf(Screen.Main) }
    var showStopConfirm by remember { mutableStateOf(false) }

    if (showStopConfirm) {
        AlertDialog(
            onDismissRequest = { showStopConfirm = false },
            containerColor = Color(0xFF1A1A2E),
            titleContentColor = Color(0xFFEF5350),
            textContentColor = Color.White.copy(alpha = 0.85f),
            title = { Text("Noetic_seed を終了しますか？") },
            text = {
                Text(
                    "プロセスが完全に終了します。再開には PC で run.bat の実行が必要です。\n\n" +
                    "「一時停止」ならスマホからいつでも再開できます。"
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showStopConfirm = false
                        vm.sendServerCommand("stop")
                        scope.launch { drawerState.close() }
                    }
                ) { Text("終了", color = Color(0xFFEF5350)) }
            },
            dismissButton = {
                TextButton(onClick = { showStopConfirm = false }) {
                    Text("キャンセル", color = Color.White.copy(alpha = 0.7f))
                }
            }
        )
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(
                drawerContainerColor = Color(0xFF1A1A2E),
            ) {
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    "noetic-seed",
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.titleLarge,
                    color = Color(0xFF4FC3F7),
                )
                Spacer(modifier = Modifier.height(8.dp))
                HorizontalDivider(color = Color(0xFF333355))

                NavigationDrawerItem(
                    label = { Text("メイン", color = Color.White) },
                    selected = currentScreen == Screen.Main,
                    onClick = {
                        currentScreen = Screen.Main
                        scope.launch { drawerState.close() }
                    },
                    colors = NavigationDrawerItemDefaults.colors(
                        selectedContainerColor = Color(0xFF2A2A4A),
                        unselectedContainerColor = Color.Transparent,
                    ),
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
                NavigationDrawerItem(
                    label = { Text("ダッシュボード", color = Color.White) },
                    selected = currentScreen == Screen.Dashboard,
                    onClick = {
                        currentScreen = Screen.Dashboard
                        scope.launch { drawerState.close() }
                    },
                    colors = NavigationDrawerItemDefaults.colors(
                        selectedContainerColor = Color(0xFF2A2A4A),
                        unselectedContainerColor = Color.Transparent,
                    ),
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
                NavigationDrawerItem(
                    label = { Text("ターミナル", color = Color.White) },
                    selected = currentScreen == Screen.Terminal,
                    onClick = {
                        currentScreen = Screen.Terminal
                        scope.launch { drawerState.close() }
                    },
                    colors = NavigationDrawerItemDefaults.colors(
                        selectedContainerColor = Color(0xFF2A2A4A),
                        unselectedContainerColor = Color.Transparent,
                    ),
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
                NavigationDrawerItem(
                    label = { Text("🧪 Test", color = Color.White) },
                    selected = currentScreen == Screen.Test,
                    onClick = {
                        currentScreen = Screen.Test
                        scope.launch { drawerState.close() }
                    },
                    colors = NavigationDrawerItemDefaults.colors(
                        selectedContainerColor = Color(0xFF2A2A4A),
                        unselectedContainerColor = Color.Transparent,
                    ),
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
                HorizontalDivider(color = Color(0xFF333355), modifier = Modifier.padding(vertical = 8.dp))
                NavigationDrawerItem(
                    label = { Text("⚙ 設定", color = Color.White) },
                    selected = currentScreen == Screen.Settings,
                    onClick = {
                        currentScreen = Screen.Settings
                        scope.launch { drawerState.close() }
                        // 設定画面を開いたら最新のプロバイダ一覧を取得
                        vm.requestLlmProviders()
                    },
                    colors = NavigationDrawerItemDefaults.colors(
                        selectedContainerColor = Color(0xFF2A2A4A),
                        unselectedContainerColor = Color.Transparent,
                    ),
                    modifier = Modifier.padding(horizontal = 12.dp),
                )
                Spacer(modifier = Modifier.weight(1f))
                HorizontalDivider(color = Color(0xFF333355), modifier = Modifier.padding(vertical = 8.dp))
                NavigationDrawerItem(
                    icon = {
                        Icon(
                            Icons.Default.PowerSettingsNew,
                            contentDescription = null,
                            tint = Color(0xFFEF5350),
                        )
                    },
                    label = { Text("Noetic_seed を終了", color = Color(0xFFEF5350)) },
                    selected = false,
                    onClick = { showStopConfirm = true },
                    colors = NavigationDrawerItemDefaults.colors(
                        unselectedContainerColor = Color.Transparent,
                    ),
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                )
                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    ) {
        val pagerState = rememberPagerState(initialPage = 0, pageCount = { 4 })

        // ドロワーのクリックとpagerを同期（Settings は pager 外なので同期対象外）
        LaunchedEffect(currentScreen) {
            val target = when (currentScreen) {
                Screen.Main -> 0
                Screen.Dashboard -> 1
                Screen.Terminal -> 2
                Screen.Test -> 3
                Screen.Settings -> null  // Settings は pager と独立
            }
            if (target != null && pagerState.currentPage != target) {
                pagerState.animateScrollToPage(target)
            }
        }
        LaunchedEffect(pagerState.currentPage) {
            // Settings 表示中にスワイプされても Settings のまま（無視）
            if (currentScreen != Screen.Settings) {
                currentScreen = when (pagerState.currentPage) {
                    0 -> Screen.Main
                    1 -> Screen.Dashboard
                    2 -> Screen.Terminal
                    else -> Screen.Test
                }
            }
        }

        if (currentScreen == Screen.Settings) {
            SettingsScreen(
                state = state,
                onMenuClick = { scope.launch { drawerState.open() } },
                onBack = { currentScreen = Screen.Main },
                onRequestProviders = { vm.requestLlmProviders() },
                onSetLlm = { provider, model, apiKey, baseUrl ->
                    vm.setLlm(provider, model, apiKey, baseUrl)
                },
                onDismissResult = { vm.clearLlmSetResult() },
            )
        } else {
            HorizontalPager(
                state = pagerState,
                modifier = Modifier.fillMaxSize(),
            ) { page ->
                when (page) {
                    0 -> MainScreen(
                        state = state,
                        onMenuClick = { scope.launch { drawerState.open() } },
                        onSendChat = { text -> vm.sendChat(text) },
                        onApproval = { id, approved -> vm.sendApproval(id, approved) },
                        onPauseToggle = {
                            vm.sendServerCommand(if (state.paused) "resume" else "pause")
                        },
                    )
                    1 -> DashboardScreen(
                        state = state,
                        onMenuClick = { scope.launch { drawerState.open() } },
                    )
                    2 -> TerminalScreen(
                        state = state,
                        onMenuClick = { scope.launch { drawerState.open() } },
                    )
                    3 -> TestScreen(
                        state = state,
                        onMenuClick = { scope.launch { drawerState.open() } },
                        onRunTool = { name, args -> vm.runTestTool(name, args) },
                    )
                }
            }
        }
    }
}

// ===== Connect Screen =====
@Composable
fun ConnectScreen(context: Context = LocalContext.current, onConnect: (String, String) -> Unit) {
    // 前回の接続情報をプリフィル（セッション TTL 超過後でも 1 タップで再接続できるように）
    val prefs = remember { context.getSharedPreferences("noetic_seed", Context.MODE_PRIVATE) }
    var url by remember { mutableStateOf(prefs.getString("ws_url", "ws://192.168.1.") ?: "ws://192.168.1.") }
    var token by remember { mutableStateOf(prefs.getString("ws_token", "") ?: "") }

    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0A0A1A)) {
        Column(
            modifier = Modifier.fillMaxSize().padding(32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("noetic-seed", style = MaterialTheme.typography.headlineLarge, color = Color(0xFF4FC3F7))
            Spacer(modifier = Modifier.height(8.dp))
            Text("entropy-driven autonomous AI", style = MaterialTheme.typography.bodyMedium, color = Color.Gray)
            Spacer(modifier = Modifier.height(48.dp))

            OutlinedTextField(
                value = url, onValueChange = { url = it },
                label = { Text("WebSocket URL") }, singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color.White, unfocusedTextColor = Color.White,
                    focusedBorderColor = Color(0xFF4FC3F7), unfocusedBorderColor = Color.Gray,
                    focusedLabelColor = Color(0xFF4FC3F7), unfocusedLabelColor = Color.Gray,
                ),
            )
            Spacer(modifier = Modifier.height(16.dp))
            OutlinedTextField(
                value = token, onValueChange = { token = it },
                label = { Text("Token") }, singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Go),
                keyboardActions = KeyboardActions(onGo = { onConnect(url, token) }),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color.White, unfocusedTextColor = Color.White,
                    focusedBorderColor = Color(0xFF4FC3F7), unfocusedBorderColor = Color.Gray,
                    focusedLabelColor = Color(0xFF4FC3F7), unfocusedLabelColor = Color.Gray,
                ),
            )
            Spacer(modifier = Modifier.height(32.dp))
            Button(
                onClick = { onConnect(url, token) },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4FC3F7)),
            ) { Text("Connect", color = Color.Black) }
        }
    }
}

// ===== Profile Select Screen =====
@Composable
fun ProfileSelectScreen(profiles: List<ProfileInfo>, onSelect: (String) -> Unit) {
    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0A0A1A)) {
        Column(
            modifier = Modifier.fillMaxSize().padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(modifier = Modifier.height(60.dp))
            Text("noetic-seed", style = MaterialTheme.typography.headlineLarge, color = Color(0xFF4FC3F7))
            Spacer(modifier = Modifier.height(8.dp))
            Text("Select Profile", style = MaterialTheme.typography.bodyMedium, color = Color.Gray)
            Spacer(modifier = Modifier.height(32.dp))

            for (profile in profiles) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 6.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(Color(0xFF1A1A2E))
                        .clickable { onSelect(profile.name) }
                        .padding(20.dp)
                ) {
                    Column {
                        Text(
                            profile.name,
                            color = Color.White,
                            fontSize = 20.sp, fontWeight = FontWeight.Bold,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            Text(
                                "cycle: ${profile.cycleId}",
                                color = Color.Gray, fontSize = 12.sp, fontFamily = FontFamily.Monospace,
                            )
                            Text(
                                "entropy: ${"%.3f".format(profile.entropy)}",
                                color = entropyToColor(profile.entropy),
                                fontSize = 12.sp, fontFamily = FontFamily.Monospace,
                            )
                            Text(
                                "energy: ${"%.0f".format(profile.energy)}",
                                color = Color.Gray, fontSize = 12.sp, fontFamily = FontFamily.Monospace,
                            )
                        }
                    }
                }
            }
        }
    }
}

// ===== Main Screen =====
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    state: IkuState,
    onMenuClick: () -> Unit,
    onSendChat: (String) -> Unit = {},
    onApproval: (String, Boolean) -> Unit = { _, _ -> },
    onPauseToggle: () -> Unit = {},
) {
    val bgColor by animateColorAsState(
        targetValue = entropyToColor(state.entropy),
        animationSpec = tween(durationMillis = 2000), label = "bg",
    )
    Box(modifier = Modifier.fillMaxSize()) {
        // 背景グラデーション（entropy連動）
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(bgColor.copy(alpha = 0.3f), Color(0xFF0A0A1A)),
                        startY = 0f, endY = 1200f,
                    )
                )
        )

        Column(modifier = Modifier.fillMaxSize()) {
            // ステータスバー
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 40.dp, start = 16.dp, end = 16.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onMenuClick) {
                    Icon(Icons.Default.Menu, "menu", tint = Color.White)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    state.selfModel["name"] ?: "noetic-seed",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White, fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.weight(1f))
                // pause/resume トグル
                IconButton(
                    onClick = onPauseToggle,
                    modifier = Modifier.size(36.dp),
                ) {
                    Icon(
                        if (state.paused) Icons.Default.PlayArrow else Icons.Default.Pause,
                        contentDescription = if (state.paused) "再開" else "一時停止",
                        tint = if (state.paused) Color(0xFFFFB74D) else Color.White.copy(alpha = 0.8f),
                    )
                }
                Spacer(modifier = Modifier.width(8.dp))
                // 接続インジケータ
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(if (state.connected) Color(0xFF76FF03) else Color.Gray)
                )
            }

            // 一時停止中バナー（paused のときのみ表示）
            if (state.paused) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 4.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFFFFB74D).copy(alpha = 0.18f))
                        .padding(horizontal = 14.dp, vertical = 10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Default.Pause,
                            contentDescription = null,
                            tint = Color(0xFFFFB74D),
                            modifier = Modifier.size(22.dp),
                        )
                        Spacer(modifier = Modifier.width(10.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "一時停止中",
                                color = Color(0xFFFFB74D),
                                fontSize = 14.sp, fontWeight = FontWeight.Bold,
                            )
                            Text(
                                "サイクルは停止、外部メッセージは受信中",
                                color = Color.White.copy(alpha = 0.65f),
                                fontSize = 11.sp,
                            )
                        }
                        Button(
                            onClick = onPauseToggle,
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFFB74D)),
                            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 14.dp, vertical = 8.dp),
                        ) {
                            Icon(
                                Icons.Default.PlayArrow,
                                contentDescription = null,
                                tint = Color.Black,
                                modifier = Modifier.size(18.dp),
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("再開", color = Color.Black, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }

            // 中央キャラ表示領域（将来3Dモデル。今はentropy表示）
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    // entropyサークル
                    Box(
                        modifier = Modifier
                            .size(180.dp)
                            .clip(CircleShape)
                            .background(bgColor.copy(alpha = 0.4f)),
                        contentAlignment = Alignment.Center,
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(
                                "${"%.3f".format(state.entropy)}",
                                fontSize = 36.sp, color = Color.White,
                                fontWeight = FontWeight.Light, fontFamily = FontFamily.Monospace,
                            )
                            Text("entropy", fontSize = 12.sp, color = Color.White.copy(alpha = 0.6f))
                        }
                    }
                    Spacer(modifier = Modifier.height(24.dp))
                    // ステータスチップス
                    Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                        StatusChip("cycle", "${state.cycleId}")
                        StatusChip("lv", "${state.toolLevel}")
                        StatusChip("energy", "${"%.0f".format(state.energy)}")
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    // E値
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        EValueChip("E1", state.e1)
                        EValueChip("E2", state.e2)
                        EValueChip("E3", state.e3)
                        EValueChip("E4", state.e4)
                    }
                }
            }

            // 承認リクエスト表示
            for (req in state.approvalRequests) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 4.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFFEF5350).copy(alpha = 0.2f))
                        .padding(12.dp)
                ) {
                    Column {
                        Text(
                            "🔐 ${req.tool} 承認待ち",
                            color = Color(0xFFEF5350),
                            fontSize = 14.sp, fontWeight = FontWeight.Bold,
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            req.preview.take(200),
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 11.sp, fontFamily = FontFamily.Monospace,
                            maxLines = 6, overflow = TextOverflow.Ellipsis,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End,
                        ) {
                            OutlinedButton(
                                onClick = { onApproval(req.id, false) },
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFEF5350)),
                            ) { Text("却下") }
                            Spacer(modifier = Modifier.width(8.dp))
                            Button(
                                onClick = { onApproval(req.id, true) },
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF76FF03)),
                            ) { Text("承認", color = Color.Black) }
                        }
                    }
                }
            }

            // pending（未対応事項）表示 — タップで展開
            if (state.pendingCount > 0) {
                var pendingExpanded by remember { mutableStateOf(false) }
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 4.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFFFF6D00).copy(alpha = 0.15f))
                        .clickable { pendingExpanded = !pendingExpanded }
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("⏳", fontSize = 16.sp)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                "未対応: ${state.pendingCount}件",
                                color = Color(0xFFFFAB40),
                                fontSize = 13.sp, fontWeight = FontWeight.Bold,
                            )
                            Spacer(modifier = Modifier.weight(1f))
                            Text(
                                if (pendingExpanded) "▲" else "▼",
                                color = Color(0xFFFFAB40), fontSize = 12.sp,
                            )
                        }
                        if (pendingExpanded && state.pendingItems.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(6.dp))
                            for (item in state.pendingItems) {
                                val typeLabel = when (item["type"]) {
                                    "user_message" -> "💬"
                                    "elyth_notification" -> "🔔"
                                    "plan_step" -> "📋"
                                    else -> "•"
                                }
                                Text(
                                    "$typeLabel ${item["content"]?.take(60) ?: ""}",
                                    color = Color.White.copy(alpha = 0.8f),
                                    fontSize = 11.sp,
                                    modifier = Modifier.padding(vertical = 2.dp),
                                    maxLines = 2, overflow = TextOverflow.Ellipsis,
                                )
                            }
                        }
                    }
                }
            }

            // AIからの返信表示（最新1件）
            if (state.replies.isNotEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 4.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFF1A1A2E).copy(alpha = 0.9f))
                        .padding(12.dp)
                ) {
                    Text(
                        state.replies.last(),
                        color = Color(0xFF4FC3F7),
                        fontSize = 13.sp,
                    )
                }
            }

            // チャット入力欄
            var chatInput by remember { mutableStateOf("") }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp)
                    .navigationBarsPadding(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                OutlinedTextField(
                    value = chatInput,
                    onValueChange = { chatInput = it },
                    placeholder = { Text("メッセージを入力...", color = Color(0xFF555577)) },
                    singleLine = true,
                    modifier = Modifier.weight(1f),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = Color(0xFF4FC3F7),
                        unfocusedBorderColor = Color(0xFF333355),
                        cursorColor = Color(0xFF4FC3F7),
                    ),
                    shape = RoundedCornerShape(24.dp),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                    keyboardActions = KeyboardActions(onSend = {
                        if (chatInput.isNotBlank()) {
                            onSendChat(chatInput.trim())
                            chatInput = ""
                        }
                    }),
                )
                if (chatInput.isNotBlank()) {
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = {
                            onSendChat(chatInput.trim())
                            chatInput = ""
                        }
                    ) {
                        Text("→", color = Color(0xFF4FC3F7), fontSize = 20.sp)
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
        }

    }
}

// ===== Dashboard Screen =====
@Composable
fun DashboardScreen(state: IkuState, onMenuClick: () -> Unit) {
    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0A0A1A)) {
        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 40.dp, start = 16.dp, end = 16.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onMenuClick) {
                    Icon(Icons.Default.Menu, "menu", tint = Color.White)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text("Dashboard", style = MaterialTheme.typography.titleLarge, color = Color(0xFF4FC3F7))
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp),
            ) {
                item { DashboardContent(state = state) }
            }
        }
    }
}

@Composable
fun DashboardContent(state: IkuState) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(24.dp)
    ) {
        DashRow("entropy", "${"%.4f".format(state.entropy)}", state.entropy)
        DashRow("energy", "${"%.1f".format(state.energy)}", state.energy / 100f)
        DashRow("pressure", "${"%.2f".format(state.pressure)}", (state.pressure / 15f).coerceIn(0f, 1f))
        DashRow("cycle", "${state.cycleId}", null)
        DashRow("tool level", "${state.toolLevel}", null)

        Spacer(modifier = Modifier.height(16.dp))
        Text("E-values", style = MaterialTheme.typography.titleSmall, color = Color.Gray)
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) {
            EValueBar("E1", state.e1)
            EValueBar("E2", state.e2)
            EValueBar("E3", state.e3)
            EValueBar("E4", state.e4)
        }

        // E値推移グラフ
        if (state.eHistory.isNotEmpty()) {
            Spacer(modifier = Modifier.height(16.dp))
            Text("E-value Trend (${state.eHistory.size} cycles)", style = MaterialTheme.typography.titleSmall, color = Color.Gray)
            Spacer(modifier = Modifier.height(8.dp))
            EValueTrendChart(history = state.eHistory)
        }

        // Disposition
        if (state.disposition.isNotEmpty()) {
            Spacer(modifier = Modifier.height(16.dp))
            Text("disposition", style = MaterialTheme.typography.titleSmall, color = Color.Gray)
            Spacer(modifier = Modifier.height(8.dp))
            for ((key, value) in state.disposition) {
                DashRow(key, "${"%.2f".format(value)}", value.coerceIn(0f, 1f))
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Text("self_model", style = MaterialTheme.typography.titleSmall, color = Color.Gray)
        Spacer(modifier = Modifier.height(8.dp))
        for ((key, value) in state.selfModel) {
            if (key == "name") {
                Text(
                    "$key: $value",
                    color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 2.dp),
                )
            } else {
                Text(
                    key, color = Color(0xFF4FC3F7), fontSize = 11.sp,
                    modifier = Modifier.padding(top = 6.dp),
                )
                Text(
                    value.take(300),
                    color = Color.White.copy(alpha = 0.85f), fontSize = 11.sp,
                    modifier = Modifier.padding(start = 8.dp, bottom = 2.dp),
                )
            }
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}

@Composable
fun EValueTrendChart(history: List<List<Float>>) {
    val colors = listOf(
        Color(0xFF4FC3F7),  // E1 - blue
        Color(0xFF76FF03),  // E2 - green
        Color(0xFFFFEB3B),  // E3 - yellow
        Color(0xFFFF6D00),  // E4 - orange
    )
    val labels = listOf("E1", "E2", "E3", "E4")

    // ラベル行
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center,
    ) {
        labels.forEachIndexed { i, label ->
            Text(
                "$label ",
                color = colors[i],
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
            )
        }
    }

    Spacer(modifier = Modifier.height(4.dp))

    // Canvas描画
    androidx.compose.foundation.Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0xFF1A1A2E))
    ) {
        if (history.size < 2) return@Canvas
        val w = size.width
        val h = size.height
        val padding = 8f
        val chartW = w - padding * 2
        val chartH = h - padding * 2

        for (eIdx in 0..3) {
            val path = androidx.compose.ui.graphics.Path()
            history.forEachIndexed { i, values ->
                val x = padding + (i.toFloat() / (history.size - 1)) * chartW
                val y = padding + (1f - values.getOrElse(eIdx) { 0f }.coerceIn(0f, 1f)) * chartH
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(
                path = path,
                color = colors[eIdx],
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f),
            )
        }
    }
}

@Composable
fun DashRow(label: String, value: String, progress: Float?) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, color = Color.Gray, fontSize = 13.sp, modifier = Modifier.width(80.dp))
        if (progress != null) {
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.weight(1f).height(4.dp).padding(horizontal = 8.dp),
                color = entropyToColor(progress),
                trackColor = Color(0xFF333355),
            )
        }
        Text(value, color = Color.White, fontSize = 13.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
fun EValueBar(label: String, value: Float) {
    val pct = (value * 100).toInt()
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, color = Color.Gray, fontSize = 11.sp)
        Spacer(modifier = Modifier.height(4.dp))
        Box(
            modifier = Modifier
                .width(40.dp)
                .height(60.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(Color(0xFF333355)),
            contentAlignment = Alignment.BottomCenter,
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(value.coerceIn(0f, 1f))
                    .background(
                        when {
                            pct >= 70 -> Color(0xFF76FF03)
                            pct >= 50 -> Color(0xFFFFEB3B)
                            else -> Color(0xFFEF5350)
                        }
                    )
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text("${pct}%", color = Color.White, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
fun StatusChip(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
        Text(label, color = Color.White.copy(alpha = 0.5f), fontSize = 10.sp)
    }
}

// ===== Test Screen =====
@Composable
fun TestScreen(state: IkuState, onMenuClick: () -> Unit, onRunTool: (String, Map<String, String>) -> Unit) {
    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0A0A1A)) {
        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 40.dp, start = 16.dp, end = 16.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onMenuClick) {
                    Icon(Icons.Default.Menu, "menu", tint = Color.White)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text("🧪 Test", style = MaterialTheme.typography.titleLarge, color = Color(0xFFFFAB40))
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp),
            ) {
                item {
                    Text(
                        "ツールをAI経由ではなく直接実行してフローをテスト",
                        color = Color.Gray, fontSize = 11.sp,
                        modifier = Modifier.padding(bottom = 16.dp),
                    )
                }

                item {
                    TestSection("📸 カメラ (stream)") {
                        TestButton("背面1枚 (frames=1)") {
                            onRunTool("camera_stream", mapOf(
                                "facing" to "back", "frames" to "1", "interval_sec" to "0.5",
                                "intent" to "test single", "expect" to "1 jpeg"
                            ))
                        }
                        TestButton("背面5枚 1秒間隔") {
                            onRunTool("camera_stream", mapOf(
                                "facing" to "back", "frames" to "5", "interval_sec" to "1.0",
                                "intent" to "test stream", "expect" to "5 jpegs"
                            ))
                        }
                        TestButton("背面10枚 0.5秒間隔") {
                            onRunTool("camera_stream", mapOf(
                                "facing" to "back", "frames" to "10", "interval_sec" to "0.5",
                                "intent" to "test stream", "expect" to "10 jpegs"
                            ))
                        }
                        TestButton("前面5枚 1秒間隔") {
                            onRunTool("camera_stream", mapOf(
                                "facing" to "front", "frames" to "5", "interval_sec" to "1.0",
                                "intent" to "test stream", "expect" to "5 jpegs"
                            ))
                        }
                    }
                }

                item {
                    TestSection("🎙 マイク") {
                        TestButton("3秒録音") {
                            onRunTool("mic_record", mapOf(
                                "duration_sec" to "3.0",
                                "intent" to "test mic 3s",
                                "message" to "テスト録音"
                            ))
                        }
                        TestButton("5秒録音 (日本語)") {
                            onRunTool("mic_record", mapOf(
                                "duration_sec" to "5.0",
                                "language" to "ja",
                                "intent" to "test mic 5s ja",
                                "message" to "テスト録音"
                            ))
                        }
                        TestButton("10秒録音 (環境音)") {
                            onRunTool("mic_record", mapOf(
                                "duration_sec" to "10.0",
                                "intent" to "test ambient 10s",
                                "message" to "テスト録音"
                            ))
                        }
                    }
                }

                item {
                    TestSection("💬 出力") {
                        TestButton("output_displayテスト") {
                            onRunTool("output_display", mapOf("content" to "テストメッセージ from test tab", "intent" to "test", "expect" to "reply appears on main"))
                        }
                    }
                }

                item {
                    TestSection("🔍 記憶") {
                        TestButton("search_memory テスト") {
                            onRunTool("search_memory", mapOf("query" to "test", "intent" to "test search", "expect" to "results"))
                        }
                        TestButton("memory_store テスト") {
                            onRunTool("memory_store", mapOf("network" to "experience", "content" to "テスト記憶 from test tab", "intent" to "test", "expect" to "stored"))
                        }
                    }
                }

                item {
                    TestSection("🌸 Elyth") {
                        TestButton("elyth_info 通知取得") {
                            onRunTool("elyth_info", mapOf("section" to "notifications", "limit" to "5", "intent" to "test", "expect" to "notification list"))
                        }
                    }
                }

                item {
                    Spacer(modifier = Modifier.height(24.dp))
                    Text("直近のログ (末尾10件):", color = Color.Gray, fontSize = 12.sp)
                    Spacer(modifier = Modifier.height(8.dp))
                    for (line in state.logLines.takeLast(10)) {
                        Text(
                            line,
                            color = if ("[test]" in line) Color(0xFFFFAB40) else Color(0xFFAAAAAA),
                            fontSize = 10.sp, fontFamily = FontFamily.Monospace,
                            modifier = Modifier.padding(vertical = 1.dp),
                        )
                    }
                    Spacer(modifier = Modifier.height(64.dp))
                }
            }
        }
    }
}

@Composable
fun TestSection(title: String, content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF1A1A2E))
            .padding(12.dp)
    ) {
        Text(title, color = Color(0xFFFFAB40), fontSize = 13.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(8.dp))
        content()
    }
}

@Composable
fun TestButton(label: String, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color(0xFF2A2A4A),
            contentColor = Color.White,
        ),
    ) {
        Text(label, fontSize = 13.sp)
    }
}

// ===== Settings Screen =====
// ドロワー経由で開く独立画面。HorizontalPager とは分離。
// 現在は LLM プロバイダ選択のみ。将来的に他の設定も追加予定。
@OptIn(ExperimentalMaterial3Api::class, androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun SettingsScreen(
    state: IkuState,
    onMenuClick: () -> Unit,
    onBack: () -> Unit,
    onRequestProviders: () -> Unit,
    onSetLlm: (provider: String, model: String, apiKey: String, baseUrl: String) -> Unit,
    onDismissResult: () -> Unit,
) {
    // 画面表示時にプロバイダ一覧を要求
    LaunchedEffect(Unit) { onRequestProviders() }

    // 選択中のプロバイダ（初期値はアクティブなもの）
    var selectedProvider by remember(state.llmActive.provider) {
        mutableStateOf(state.llmActive.provider)
    }
    // モデル入力欄（選択プロバイダの last_model を初期値に）
    var modelInput by remember(selectedProvider) {
        val defaultModel = state.llmProviders.find { it.provider == selectedProvider }?.lastModel
            ?: state.llmActive.model
        mutableStateOf(defaultModel)
    }
    // API Key 入力欄（空のままなら既存温存）
    var apiKeyInput by remember(selectedProvider) { mutableStateOf("") }
    var baseUrlInput by remember(selectedProvider) {
        val defaultUrl = state.llmProviders.find { it.provider == selectedProvider }?.baseUrl ?: ""
        mutableStateOf(defaultUrl)
    }

    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0A0A1A)) {
        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 40.dp, start = 16.dp, end = 16.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onMenuClick) {
                    Icon(Icons.Default.Menu, "menu", tint = Color.White)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text("⚙ 設定", style = MaterialTheme.typography.titleLarge, color = Color(0xFF4FC3F7))
                Spacer(modifier = Modifier.weight(1f))
                TextButton(onClick = onBack) {
                    Text("戻る", color = Color(0xFF4FC3F7), fontSize = 13.sp)
                }
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp),
            ) {
                item {
                    Text(
                        "LLM プロバイダ選択",
                        color = Color(0xFF4FC3F7),
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(bottom = 8.dp),
                    )
                    Text(
                        "現在: ${state.llmActive.provider} / ${state.llmActive.model}",
                        color = Color.Gray, fontSize = 12.sp,
                        modifier = Modifier.padding(bottom = 12.dp),
                    )
                }

                // プロバイダ選択チップ
                item {
                    SettingsSection("プロバイダ") {
                        if (state.llmProviders.isEmpty()) {
                            Text("(読み込み中...)", color = Color.Gray, fontSize = 12.sp)
                        } else {
                            androidx.compose.foundation.layout.FlowRow(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                for (p in state.llmProviders) {
                                    val isSelected = selectedProvider == p.provider
                                    val isActive = state.llmActive.provider == p.provider
                                    Surface(
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(20.dp))
                                            .clickable { selectedProvider = p.provider }
                                            .padding(vertical = 4.dp),
                                        color = when {
                                            isSelected -> Color(0xFF4FC3F7)
                                            isActive -> Color(0xFF2A4A5A)
                                            else -> Color(0xFF2A2A4A)
                                        },
                                    ) {
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp)
                                        ) {
                                            Text(
                                                p.provider,
                                                color = if (isSelected) Color.Black else Color.White,
                                                fontSize = 13.sp,
                                            )
                                            if (p.hasKey) {
                                                Spacer(modifier = Modifier.width(4.dp))
                                                Text("✓", color = if (isSelected) Color.Black else Color(0xFF4CAF50), fontSize = 11.sp)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                item {
                    SettingsSection("モデル名") {
                        OutlinedTextField(
                            value = modelInput,
                            onValueChange = { modelInput = it },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("例: gemma-4-26b-a4b-it", color = Color.Gray) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF4FC3F7),
                                unfocusedBorderColor = Color(0xFF333355),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                cursorColor = Color(0xFF4FC3F7),
                            ),
                            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 13.sp),
                            singleLine = true,
                        )
                    }
                }

                item {
                    SettingsSection("API Key") {
                        val hasExisting = state.llmProviders.find { it.provider == selectedProvider }?.hasKey ?: false
                        Text(
                            if (hasExisting) "保存済み（空欄のままなら既存を使用）"
                            else "未登録（新規入力）",
                            color = if (hasExisting) Color(0xFF4CAF50) else Color(0xFFFFAB40),
                            fontSize = 11.sp,
                            modifier = Modifier.padding(bottom = 4.dp),
                        )
                        OutlinedTextField(
                            value = apiKeyInput,
                            onValueChange = { apiKeyInput = it },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("sk-... or 空欄", color = Color.Gray) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF4FC3F7),
                                unfocusedBorderColor = Color(0xFF333355),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                cursorColor = Color(0xFF4FC3F7),
                            ),
                            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 13.sp),
                            singleLine = true,
                            visualTransformation = androidx.compose.ui.text.input.PasswordVisualTransformation(),
                        )
                    }
                }

                item {
                    SettingsSection("Base URL（任意）") {
                        OutlinedTextField(
                            value = baseUrlInput,
                            onValueChange = { baseUrlInput = it },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("空欄ならデフォルト", color = Color.Gray) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF4FC3F7),
                                unfocusedBorderColor = Color(0xFF333355),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                cursorColor = Color(0xFF4FC3F7),
                            ),
                            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp),
                            singleLine = true,
                        )
                    }
                }

                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(
                        onClick = {
                            onSetLlm(selectedProvider, modelInput, apiKeyInput, baseUrlInput)
                        },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF4FC3F7),
                            contentColor = Color.Black,
                        ),
                        enabled = selectedProvider.isNotBlank() && modelInput.isNotBlank(),
                    ) {
                        Text("適用（次サイクルから反映）", fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    }
                }

                item {
                    val result = state.llmSetResult
                    if (result != null) {
                        Spacer(modifier = Modifier.height(12.dp))
                        Surface(
                            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)),
                            color = if (result.startsWith("エラー")) Color(0xFF5A2A2A) else Color(0xFF2A5A2A),
                        ) {
                            Row(
                                modifier = Modifier.padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Text(
                                    result,
                                    color = Color.White,
                                    fontSize = 12.sp,
                                    modifier = Modifier.weight(1f),
                                )
                                TextButton(onClick = onDismissResult) {
                                    Text("×", color = Color.White, fontSize = 16.sp)
                                }
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(64.dp))
                }
            }
        }
    }
}

@Composable
fun SettingsSection(title: String, content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF1A1A2E))
            .padding(12.dp)
    ) {
        Text(title, color = Color(0xFF4FC3F7), fontSize = 13.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(8.dp))
        content()
    }
}

// ===== PreLaunch Config Screen =====
// プロファイル選択 → main.py 起動前の LLM 設定画面。
// SettingsScreen と内容はほぼ同じだが、確定ボタンが「起動」で、戻るは「プロファイル選択に戻る」。
@OptIn(ExperimentalMaterial3Api::class, androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun PreLaunchConfigScreen(
    profileName: String,
    state: IkuState,
    onRequestProviders: () -> Unit,
    onCancel: () -> Unit,
    onConfirm: (provider: String, model: String, apiKey: String, baseUrl: String) -> Unit,
) {
    LaunchedEffect(Unit) { onRequestProviders() }

    var selectedProvider by remember(state.llmActive.provider, state.llmProviders.size) {
        mutableStateOf(state.llmActive.provider.ifBlank {
            state.llmProviders.firstOrNull()?.provider ?: ""
        })
    }
    var modelInput by remember(selectedProvider, state.llmProviders.size) {
        val defaultModel = state.llmProviders.find { it.provider == selectedProvider }?.lastModel
            ?: state.llmActive.model
        mutableStateOf(defaultModel)
    }
    var apiKeyInput by remember(selectedProvider) { mutableStateOf("") }
    var baseUrlInput by remember(selectedProvider, state.llmProviders.size) {
        val defaultUrl = state.llmProviders.find { it.provider == selectedProvider }?.baseUrl ?: ""
        mutableStateOf(defaultUrl)
    }

    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0A0A1A)) {
        Column(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 48.dp, start = 24.dp, end = 24.dp, bottom = 12.dp),
            ) {
                Text(
                    "プロファイル起動",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color(0xFF4FC3F7),
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    profileName,
                    style = MaterialTheme.typography.headlineSmall,
                    color = Color.White,
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    "起動する LLM を選んでください",
                    color = Color.Gray,
                    fontSize = 12.sp,
                )
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp),
            ) {
                item {
                    SettingsSection("プロバイダ") {
                        if (state.llmProviders.isEmpty()) {
                            Text("(読み込み中...)", color = Color.Gray, fontSize = 12.sp)
                        } else {
                            androidx.compose.foundation.layout.FlowRow(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                for (p in state.llmProviders) {
                                    val isSelected = selectedProvider == p.provider
                                    Surface(
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(20.dp))
                                            .clickable { selectedProvider = p.provider }
                                            .padding(vertical = 4.dp),
                                        color = if (isSelected) Color(0xFF4FC3F7) else Color(0xFF2A2A4A),
                                    ) {
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp)
                                        ) {
                                            Text(
                                                p.provider,
                                                color = if (isSelected) Color.Black else Color.White,
                                                fontSize = 13.sp,
                                            )
                                            if (p.hasKey) {
                                                Spacer(modifier = Modifier.width(4.dp))
                                                Text(
                                                    "✓",
                                                    color = if (isSelected) Color.Black else Color(0xFF4CAF50),
                                                    fontSize = 11.sp,
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                item {
                    SettingsSection("モデル名") {
                        OutlinedTextField(
                            value = modelInput,
                            onValueChange = { modelInput = it },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("例: gemma-4-26b-a4b-it", color = Color.Gray) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF4FC3F7),
                                unfocusedBorderColor = Color(0xFF333355),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                cursorColor = Color(0xFF4FC3F7),
                            ),
                            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 13.sp),
                            singleLine = true,
                        )
                    }
                }

                item {
                    SettingsSection("API Key") {
                        val hasExisting = state.llmProviders.find { it.provider == selectedProvider }?.hasKey ?: false
                        Text(
                            if (hasExisting) "保存済み（空欄のままなら既存を使用）"
                            else "未登録（lmstudio以外は入力必須）",
                            color = if (hasExisting) Color(0xFF4CAF50) else Color(0xFFFFAB40),
                            fontSize = 11.sp,
                            modifier = Modifier.padding(bottom = 4.dp),
                        )
                        OutlinedTextField(
                            value = apiKeyInput,
                            onValueChange = { apiKeyInput = it },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("sk-... or 空欄", color = Color.Gray) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF4FC3F7),
                                unfocusedBorderColor = Color(0xFF333355),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                cursorColor = Color(0xFF4FC3F7),
                            ),
                            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 13.sp),
                            singleLine = true,
                            visualTransformation = androidx.compose.ui.text.input.PasswordVisualTransformation(),
                        )
                    }
                }

                item {
                    SettingsSection("Base URL（任意）") {
                        OutlinedTextField(
                            value = baseUrlInput,
                            onValueChange = { baseUrlInput = it },
                            modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("空欄ならデフォルト", color = Color.Gray) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF4FC3F7),
                                unfocusedBorderColor = Color(0xFF333355),
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                cursorColor = Color(0xFF4FC3F7),
                            ),
                            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 12.sp),
                            singleLine = true,
                        )
                    }
                }

                item {
                    Spacer(modifier = Modifier.height(20.dp))
                    Button(
                        onClick = {
                            onConfirm(selectedProvider, modelInput, apiKeyInput, baseUrlInput)
                        },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF4FC3F7),
                            contentColor = Color.Black,
                        ),
                        enabled = selectedProvider.isNotBlank() && modelInput.isNotBlank(),
                    ) {
                        Text("この設定で起動", fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    OutlinedButton(
                        onClick = onCancel,
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFAAAAAA)),
                    ) {
                        Text("プロファイル選択に戻る", fontSize = 13.sp)
                    }
                    Spacer(modifier = Modifier.height(64.dp))
                }
            }
        }
    }
}

// ===== Terminal Screen =====
@Composable
fun TerminalScreen(state: IkuState, onMenuClick: () -> Unit) {
    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0A0A1A)) {
        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 40.dp, start = 16.dp, end = 16.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onMenuClick) {
                    Icon(Icons.Default.Menu, "menu", tint = Color.White)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text("Terminal", style = MaterialTheme.typography.titleLarge, color = Color(0xFF4FC3F7))
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    if (state.connected) "● live" else "○ offline",
                    color = if (state.connected) Color(0xFF76FF03) else Color.Gray,
                    fontSize = 12.sp,
                )
            }

            val listState = rememberLazyListState()
            LaunchedEffect(state.logLines.size) {
                if (state.logLines.isNotEmpty()) {
                    listState.animateScrollToItem(state.logLines.size - 1)
                }
            }

            LazyColumn(
                state = listState,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            ) {
                items(state.logLines) { line ->
                    val color = when {
                        line.startsWith("---") -> Color(0xFF4FC3F7)
                        "エラー" in line || "失敗" in line -> Color(0xFFEF5350)
                        "[pressure]" in line -> Color(0xFF555577)
                        "選択:" in line -> Color(0xFF76FF03)
                        "実行:" in line -> Color(0xFFFFEB3B)
                        else -> Color(0xFFAAAAAAA)
                    }
                    Text(
                        text = line, color = color,
                        fontSize = 10.sp, fontFamily = FontFamily.Monospace,
                        maxLines = 4, overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.padding(vertical = 1.dp),
                    )
                }
            }
        }
    }
}

@Composable
fun EValueChip(label: String, value: Float) {
    val pct = (value * 100).toInt()
    val color = when {
        pct >= 70 -> Color(0xFF76FF03)
        pct >= 50 -> Color(0xFFFFEB3B)
        else -> Color(0xFFEF5350)
    }
    Text(
        "$label:${pct}%", color = color,
        fontSize = 12.sp, fontFamily = FontFamily.Monospace,
    )
}
