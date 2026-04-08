import type { User } from "@/types";

const TOKEN_KEY = "ragent_token";
const USER_KEY = "ragent_user";
const THEME_KEY = "ragent_theme";
/** 聊天 RAG 检索时使用的知识库 ID（在「知识库文档」页进入时自动写入） */
const ACTIVE_KB_ID_KEY = "ragent_active_kb_id";

function safeGet(key: string) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    return;
  }
}

function safeRemove(key: string) {
  try {
    window.localStorage.removeItem(key);
  } catch {
    return;
  }
}

export const storage = {
  getToken(): string | null {
    return safeGet(TOKEN_KEY);
  },
  setToken(token: string) {
    safeSet(TOKEN_KEY, token);
  },
  clearToken() {
    safeRemove(TOKEN_KEY);
  },
  getUser(): User | null {
    const raw = safeGet(USER_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as User;
    } catch {
      return null;
    }
  },
  setUser(user: User) {
    safeSet(USER_KEY, JSON.stringify(user));
  },
  clearUser() {
    safeRemove(USER_KEY);
  },
  clearAuth() {
    safeRemove(TOKEN_KEY);
    safeRemove(USER_KEY);
  },
  getTheme(): string | null {
    return safeGet(THEME_KEY);
  },
  setTheme(theme: string) {
    safeSet(THEME_KEY, theme);
  },
  getActiveKnowledgeBaseId(): string | null {
    return safeGet(ACTIVE_KB_ID_KEY);
  },
  setActiveKnowledgeBaseId(kbId: string) {
    safeSet(ACTIVE_KB_ID_KEY, kbId);
  },
  clearActiveKnowledgeBaseId() {
    safeRemove(ACTIVE_KB_ID_KEY);
  }
};
