package com.example.noetic_seed_monitor

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
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
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
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
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // ナビゲーションバーを隠す（スワイプで一時表示）
        val controller = androidx.core.view.WindowCompat.getInsetsController(window, window.decorView)
        controller.hide(androidx.core.view.WindowInsetsCompat.Type.navigationBars())
        controller.systemBarsBehavior =
            androidx.core.view.WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE

        setContent {
            IkuApp()
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
enum class Screen { Main, Terminal }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun IkuApp(vm: IkuViewModel = viewModel()) {
    val state by vm.state.collectAsState()
    var hasConnected by remember { mutableStateOf(false) }
    if (state.connected) hasConnected = true

    if (!hasConnected) {
        ConnectScreen(onConnect = { url, token -> vm.connect(url, token) })
    } else {
        AppWithDrawer(state = state, vm = vm)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppWithDrawer(state: IkuState, vm: IkuViewModel) {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    var currentScreen by remember { mutableStateOf(Screen.Main) }

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
                // 将来: 承認、設定
            }
        }
    ) {
        when (currentScreen) {
            Screen.Main -> MainScreen(
                state = state,
                onMenuClick = { scope.launch { drawerState.open() } },
                onSendChat = { text -> vm.sendChat(text) },
            )
            Screen.Terminal -> TerminalScreen(
                state = state,
                onMenuClick = { scope.launch { drawerState.open() } },
            )
        }
    }
}

// ===== Connect Screen =====
@Composable
fun ConnectScreen(onConnect: (String, String) -> Unit) {
    var url by remember { mutableStateOf("ws://192.168.1.") }
    var token by remember { mutableStateOf("") }

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

// ===== Main Screen =====
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(state: IkuState, onMenuClick: () -> Unit, onSendChat: (String) -> Unit = {}) {
    val bgColor by animateColorAsState(
        targetValue = entropyToColor(state.entropy),
        animationSpec = tween(durationMillis = 2000), label = "bg",
    )
    val sheetState = rememberModalBottomSheetState()
    var showDashboard by remember { mutableStateOf(false) }

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
                // 接続インジケータ
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(if (state.connected) Color(0xFF76FF03) else Color.Gray)
                )
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

        // ダッシュボードFAB
        FloatingActionButton(
            onClick = { showDashboard = true },
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(end = 16.dp, bottom = 80.dp),
            containerColor = Color(0xFF4FC3F7).copy(alpha = 0.8f),
            contentColor = Color.Black,
        ) {
            Icon(Icons.Default.BarChart, "dashboard")
        }

        // ダッシュボード BottomSheet
        if (showDashboard) {
            ModalBottomSheet(
                onDismissRequest = { showDashboard = false },
                sheetState = sheetState,
                containerColor = Color(0xFF1A1A2E).copy(alpha = 0.95f),
            ) {
                DashboardContent(state = state)
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
        Text("Dashboard", style = MaterialTheme.typography.titleMedium, color = Color(0xFF4FC3F7))
        Spacer(modifier = Modifier.height(16.dp))

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

        Spacer(modifier = Modifier.height(16.dp))
        Text("self_model", style = MaterialTheme.typography.titleSmall, color = Color.Gray)
        Spacer(modifier = Modifier.height(8.dp))
        for ((key, value) in state.selfModel) {
            Text(
                "$key: ${value.take(80)}",
                color = Color.White, fontSize = 12.sp, fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(vertical = 2.dp),
            )
        }

        Spacer(modifier = Modifier.height(32.dp))
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
