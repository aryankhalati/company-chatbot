import "dotenv/config";
import { ChatGroq } from "@langchain/groq";

const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "openai/gpt-oss-120b",
  temperature: 0, // 0 = steady, least creative answers (good for fact-based bots)
});

const reply = await llm.invoke("Reply with exactly: LLM is working");
console.log(reply.content);