/**
 * Gemini AI Service for Movie & Series Insights
 * Clean, streamlined implementation using Gemini API
 */

import { GoogleGenAI } from "@google/genai";
import { getLogger } from "./logger";

interface MovieData {
  title?: string;
  name?: string; // For TV series
  release_date?: string;
  first_air_date?: string;
  overview?: string;
  genre_ids?: number[];
  vote_average?: number;
  id?: number;
  director?: string;
  cast?: string[];
  runtime?: number;
  genres?: { id?: number; name: string }[];
  production_companies?: { name: string }[];
  original_language?: string;

  // TV-specific fields
  created_by?: { name: string }[];
  episode_run_time?: number[];
  number_of_seasons?: number;
  number_of_episodes?: number;
  networks?: { name: string }[];
}

interface AIFactsResponse {
  facts: string[];
  success: boolean;
  error?: string;
}

interface AISuggestionResponse {
  suggestion: {
    title: string;
    year: string;
    type: "movie" | "series";
    overview: string;
    reason: string;
    searchKeyword: string;
  };
  success: boolean;
  error?: string;
}

class GeminiService {
  private ai: GoogleGenAI;
  private logger = getLogger();
  //private readonly MODEL_NAME = "gemini-2.5-flash";
  //private readonly MODEL_NAME = "gemini-1.5-flash";
private readonly MODEL_NAME = "gemini-2.0-flash";


  constructor(apiKey: string) {
    if (!apiKey?.trim()) {
      throw new Error("Invalid API key provided");
    }

    this.ai = new GoogleGenAI({ apiKey });
    this.logger.info("GeminiService initialized successfully");
  }

  // ------------------------------------------
  // PROMPT GENERATION
  // ------------------------------------------

  private createMoviePrompt(data: MovieData): string {
    const isTV = !!data.name;
    const title = isTV ? data.name : data.title;

    const releaseYear = isTV
      ? data.first_air_date
        ? new Date(data.first_air_date).getFullYear()
        : "Unknown"
      : data.release_date
        ? new Date(data.release_date).getFullYear()
        : "Unknown";

    return `You are an expert cinematic researcher with deep knowledge of movies and TV shows.

GOAL: Generate Top 10 most exciting and factual information about "${title}" (${releaseYear}) - a ${isTV ? "TV Series" : "Movie"}.

IMPORTANT: Use only your training data knowledge. Focus on well-known, verifiable facts that would be commonly reported.

Basic Info:
- Overview: ${data.overview || "Not available"}
- Language: ${data.original_language || "en"}
- Genres: ${data.genres?.map(g => g.name).join(", ") || "various"}
- Runtime: ${data.runtime ? `${data.runtime} minutes` : "not specified"}
${data.production_companies?.length ? `- Producers: ${data.production_companies.map(pc => pc.name).join(", ")}` : ""}
${isTV && data.created_by?.length ? `- Creators: ${data.created_by.map(c => c.name).join(", ")}` : ""}
${isTV && data.networks?.length ? `- Networks: ${data.networks.map(n => n.name).join(", ")}` : ""}
${isTV && data.number_of_seasons ? `- Seasons: ${data.number_of_seasons}, Episodes: ${data.number_of_episodes}` : ""}

Requirements:
1. Focus on widely-known, well-documented facts from your training data
2. Format as compelling, single-paragraph facts
3. Include box office/streaming performance if available
4. Cover production stories, cast details, awards, controversies
5. Make each fact attention-grabbing and viral-worthy
6. Ensure accuracy - only include information you're confident about

Return exactly 10 facts as JSON array:
["Exciting fact 1", "Exciting fact 2", ...]

Return ONLY the JSON array, no other text.`;
  }

  private createSuggestionPrompt(): string {
    return `You are a film curator recommending hidden gems and underrated titles.

Recommend one exceptional movie or TV series that deserves more recognition.

Focus on:
- Hidden gems and cult classics
- International and independent films
- Critically acclaimed but overlooked titles

Return as JSON:
{
  "title": "Exact title",
  "year": "Release year",
  "type": "movie or series",
  "overview": "2-3 sentence plot summary",
  "reason": "Why this is worth watching",
  "searchKeyword": "Best search term"
}

Return ONLY the JSON object, no other text.`;
  }

  // ------------------------------------------
  // PARSERS
  // ------------------------------------------

  private parseFactsResponse(responseText: string): string[] {
    const cleaned = responseText
      .replace(/^```json\s*/i, "")
      .replace(/```\s*$/i, "")
      .trim();

    try {
      const facts = JSON.parse(cleaned);
      if (Array.isArray(facts)) {
        return facts
          .filter(f => typeof f === "string" && f.trim().length > 10)
          .slice(0, 10);
      }
    } catch {
      return responseText
        .split("\n")
        .map(l => l.trim())
        .filter(l => l.length > 20 && !l.startsWith("{") && !l.startsWith("["))
        .map(l => l.replace(/^\d+\.\s*/, "").replace(/^[-*•]\s*/, ""))
        .slice(0, 10);
    }

    return [];
  }

  private parseSuggestionResponse(responseText: string) {
    const cleaned = responseText
      .replace(/^```json\s*/i, "")
      .replace(/```\s*$/i, "")
      .trim();

    const s = JSON.parse(cleaned);

    if (!s?.title || !s?.year || !s?.type || !s?.overview || !s?.reason || !s?.searchKeyword) {
      throw new Error("Invalid suggestion format");
    }

    if (s.type !== "movie" && s.type !== "series") {
      throw new Error("Invalid suggestion type");
    }

    return {
      title: s.title.trim(),
      year: s.year.toString(),
      type: s.type,
      overview: s.overview.trim(),
      reason: s.reason.trim(),
      searchKeyword: s.searchKeyword.trim()
    };
  }

  // ------------------------------------------
  // ERROR HANDLING
  // ------------------------------------------

  private handleError(error: Error): string {
    const message = error.message.toLowerCase();

    if (message.includes("api key") || message.includes("authentication")) {
      return "AI service authentication failed. Please check configuration.";
    }

    if (message.includes("quota") || message.includes("rate limit")) {
      return "AI service rate limit exceeded. Please try again later.";
    }

    if (message.includes("overloaded") || message.includes("unavailable")) {
      return "The AI service is temporarily overloaded. Please try again shortly.";
    }

    if (message.includes("blocked") || message.includes("safety")) {
      return "Content was blocked by safety filters.";
    }

    return error.message;
  }

  // ------------------------------------------
  // RETRY WRAPPER
  // ------------------------------------------

  private async retry<T>(fn: () => Promise<T>, retries = 2, delayMs = 900): Promise<T> {
    try {
      return await fn();
    } catch (err) {
      if (retries <= 0) throw err;
      await new Promise(res => setTimeout(res, delayMs));
      return this.retry(fn, retries - 1, delayMs * 2);
    }
  }

  // ------------------------------------------
  // API METHODS
  // ------------------------------------------

  async generateSuggestion(): Promise<AISuggestionResponse> {
    try {
      this.logger.info("Generating AI suggestion");

      const response = await this.ai.models.generateContent({
        model: this.MODEL_NAME,
        contents: this.createSuggestionPrompt(),
      });

      if (!response?.text) throw new Error("Empty response from Gemini API");

      const suggestion = this.parseSuggestionResponse(response.text);

      return { suggestion, success: true };
    } catch (error) {
      const msg = this.handleError(error as Error);
      this.logger.error("Error generating AI suggestion", { error: msg });

      return {
        suggestion: {
          title: "",
          year: "",
          type: "movie",
          overview: "",
          reason: "",
          searchKeyword: "",
        },
        success: false,
        error: msg,
      };
    }
  }

  async generateFacts(data: MovieData): Promise<AIFactsResponse> {
    try {
      const title = data.name || data.title;
      if (!title) throw new Error("Title is required");

      this.logger.info("Generating AI facts", { title });

      const response = await this.retry(() =>
        this.ai.models.generateContent({
          model: this.MODEL_NAME,
          contents: this.createMoviePrompt(data),
        })
      );

      if (!response?.text) throw new Error("Empty response from Gemini API");

      const facts = this.parseFactsResponse(response.text);
      if (facts.length === 0) throw new Error("No valid facts extracted");

      return { facts, success: true };
    } catch (error) {
      const msg = this.handleError(error as Error);
      this.logger.error("Error generating AI facts", { error: msg });

      return { facts: [], success: false, error: msg };
    }
  }
}

// ------------------------------------------
// SINGLETON EXPORT
// ------------------------------------------

let geminiServiceInstance: GeminiService | null = null;

export function getGeminiService(): GeminiService {
  if (!geminiServiceInstance) {
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey?.trim()) {
      const logger = getLogger();
      logger.error("GEMINI_API_KEY is missing");
      throw new Error("GEMINI_API_KEY is not configured");
    }

    geminiServiceInstance = new GeminiService(apiKey);
  }

  return geminiServiceInstance;
}

export type { MovieData, AIFactsResponse, AISuggestionResponse };
