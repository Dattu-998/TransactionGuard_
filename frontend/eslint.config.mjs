import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: { globals: { window: "readonly", document: "readonly", performance: "readonly", navigator: "readonly", URL: "readonly", Blob: "readonly", FormData: "readonly", fetch: "readonly", AbortController: "readonly", KeyboardEvent: "readonly", HTMLInputElement: "readonly" } },
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { varsIgnorePattern: "^_", argsIgnorePattern: "^_", ignoreRestSiblings: true }],
    },
  },
);
