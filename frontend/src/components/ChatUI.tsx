"use client";

import { useChat, Message } from "@/hooks/useChat";
import { useRef, useEffect, useState } from "react";
import {
  Send,
  Loader2,
  Bot,
  User,
  Trash2,
  StopCircle,
  Sparkles,
} from "lucide-react";

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser
            ? "bg-blue-500 text-white"
            : "bg-gradient-to-br from-purple-500 to-indigo-600 text-white"
        }`}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>
      <div
        className={`flex-1 max-w-[80%] ${
          isUser ? "items-end" : "items-start"
        }`}
      >
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? "bg-blue-500 text-white rounded-tr-md"
              : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-tl-md"
          }`}
        >
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
            )}
          </p>
        </div>
        <span className="text-xs text-gray-400 mt-1 block">
          {message.createdAt?.toLocaleTimeString("zh-CN", {
            hour: "2-digit",
            minute: "2-digit",
          }) || ""}
        </span>
      </div>
    </div>
  );
}

export default function ChatUI() {
  const {
    messages,
    isLoading,
    sendMessage,
    stopStreaming,
    clearMessages,
  } = useChat();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [inputValue, setInputValue] = useState("");

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const input = inputRef.current;
    if (!input || !inputValue.trim() || isLoading) return;

    const message = inputValue.trim();
    setInputValue("");
    input.value = "";
    input.style.height = "auto";
    await sendMessage(message);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInputResize = () => {
    const input = inputRef.current;
    if (input) {
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
    }
  };

  return (
    <div className="flex flex-col h-screen bg-white dark:bg-gray-900">
      <header className="flex-shrink-0 border-b border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
              <Sparkles className="text-white" size={20} />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
                智能扫地机器人客服
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                24小时在线为您服务
              </p>
            </div>
          </div>
          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
              title="清空对话"
            >
              <Trash2 size={20} />
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center mb-6">
                <Bot className="text-white" size={40} />
              </div>
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-2">
                您好，我是智能客服助手
              </h2>
              <p className="text-gray-500 dark:text-gray-400 mb-8 max-w-md">
                我可以帮您解答扫地机器人的使用问题、故障排除、维护保养等问题。
              </p>
              <div className="grid grid-cols-2 gap-3 w-full max-w-md">
                {[
                  "如何清洁扫地机器人？",
                  "扫地机器人无法启动怎么办？",
                  "如何生成使用报告？",
                  "扫地机器人的保养方法",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => {
                      setInputValue(suggestion);
                      inputRef.current?.focus();
                    }}
                    className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-xl transition-colors text-left"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      <footer className="flex-shrink-0 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-end gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => {
                  setInputValue(e.target.value);
                  handleInputResize();
                }}
                onKeyDown={handleKeyDown}
                placeholder="输入您的问题..."
                disabled={isLoading}
                rows={1}
                className="w-full resize-none rounded-2xl border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 px-4 py-3 pr-12 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:opacity-50 transition-all"
              />
            </div>
            {isLoading ? (
              <button
                type="button"
                onClick={stopStreaming}
                className="flex-shrink-0 p-3 bg-red-500 hover:bg-red-600 text-white rounded-full transition-colors"
                title="停止生成"
              >
                <StopCircle size={20} />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!inputValue.trim()}
                className="flex-shrink-0 p-3 bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white rounded-full transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send size={20} />
              </button>
            )}
          </div>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-2 text-center">
            按 Enter 发送，Shift + Enter 换行
          </p>
        </form>
      </footer>
    </div>
  );
}
