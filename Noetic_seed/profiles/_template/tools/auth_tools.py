"""auth_profile_info — 認証プロファイルのメタ情報アクセス
機密情報（token/key/secret 等）は返さない。iku が auth_profile の存在と
メタ情報（type, app_id, endpoint 等）を知るためのツール。"""
import json as _json
from core.runtime.registry import ToolError, normalize_detail, result_redactor
from core.auth import get_auth_profile_info, list_auth_profile_names


def auth_profile_info(args: dict) -> str:
    """auth_profile のメタ情報取得。
    - name 未指定: 登録プロファイル名の一覧
    - name 指定: そのプロファイルのメタ情報（機密フィールドを除く）"""
    name = str(args.get("name", "")).strip()

    if not name:
        # 一覧モード
        try:
            names = list_auth_profile_names(strict=True)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError("エラー: 認証情報の取得失敗", detail={"exception_type": type(e).__name__}) from None
        if not names:
            return "[auth_profile_info] 登録されたプロファイルはありません"
        return result_redactor()(f"[auth_profile_info] 登録プロファイル: {', '.join(names)}")

    # 詳細モード
    try:
        info = get_auth_profile_info(name, strict=True)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError("エラー: 認証情報の取得失敗", detail={"exception_type": type(e).__name__}) from None
    if info is None:
        raise ToolError(result_redactor()(f"エラー: auth profile '{name}' は存在しません"))

    redact = result_redactor()
    info, detail_error = normalize_detail(info, redact)
    if detail_error:
        raise ToolError("エラー: 認証情報の取得失敗", detail={"detail_error": detail_error})
    name = redact(name)
    text = _json.dumps(info, ensure_ascii=False, indent=2)
    return f"[auth_profile_info] {name}\n{text}\n\n※ token/key/secret 等の機密フィールドは表示されません。http_request の auth= で参照して使ってください。"
