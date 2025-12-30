import { existsSync, readFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";

export const CODEX_PROVIDER = "codex";
export const CODEX_ORIGINATOR = "codex_cli_rs";

export interface CodexAuthTokens {
	accessToken: string;
	accountId: string;
}

interface CodexAuthFile {
	tokens?: {
		access_token?: string;
		account_id?: string;
	};
}

export function getCodexHome(): string {
	return process.env.CODEX_HOME || join(homedir(), ".codex");
}

export function getCodexAuthPath(codexHome: string = getCodexHome()): string {
	return join(codexHome, "auth.json");
}

export function loadCodexAuth(codexHome?: string): CodexAuthTokens {
	const authPath = getCodexAuthPath(codexHome);
	if (!existsSync(authPath)) {
		throw new Error(`Codex auth file not found at ${authPath}. Run "codex login" or set CODEX_HOME.`);
	}

	let raw: string;
	try {
		raw = readFileSync(authPath, "utf-8");
	} catch (error) {
		throw new Error(`Failed to read Codex auth file at ${authPath}: ${String(error)}`);
	}

	let parsed: CodexAuthFile;
	try {
		parsed = JSON.parse(raw) as CodexAuthFile;
	} catch (error) {
		throw new Error(`Failed to parse Codex auth file at ${authPath}: ${String(error)}`);
	}

	const accessToken = parsed.tokens?.access_token;
	const accountId = parsed.tokens?.account_id;
	if (!accessToken || !accountId) {
		throw new Error(`Codex auth file missing tokens.access_token or tokens.account_id: ${authPath}`);
	}

	return { accessToken, accountId };
}

export function findCodexPromptFile(): string | null {
	const repoEnv = process.env.CODEX_REPO;
	const locations = [
		repoEnv,
		"/workspace/openai/codex",
		join(homedir(), "codex"),
		join(homedir(), "src/codex"),
		"/opt/codex",
	].filter((dir): dir is string => Boolean(dir));

	for (const dir of locations) {
		const candidate = join(dir, "codex-rs", "core", "gpt-5.2-codex_prompt.md");
		if (existsSync(candidate)) {
			return candidate;
		}
	}

	return null;
}
