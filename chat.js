import "dotenv/config";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";

const CUTOFF = 0.6; // below this score, a chunk counts as "not relevant"

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
  apiKey: process.env.GOOGLE_API_KEY,
});

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.Index(process.env.PINECONE_INDEX);
const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex: index,
});

const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "openai/gpt-oss-120b",
  temperature: 0,
});

const NO_ANSWER = "I don't have that.";

const SYSTEM_PROMPT = `You answer questions about the CodersGyan company wiki.
Use ONLY the context below. Do not use outside knowledge.
If the context does not contain the answer, reply exactly: ${NO_ANSWER}
Keep the answer short and clear.`;

// Guard 1: retrieval with a score cutoff
async function retrieve(question) {
  const results = await vectorStore.similaritySearchWithScore(question, 3);
  return results.filter(([doc, score]) => score >= CUTOFF);
}

async function ask(question) {
  console.log(`\nQuestion: ${question}`);
  const relevant = await retrieve(question);

  if (relevant.length === 0) {
    console.log(`Answer: ${NO_ANSWER} (no chunk passed the cutoff, LLM not called)`);
    return;
  }

  const context = relevant.map(([doc]) => doc.pageContent).join("\n---\n");

  // Guard 2: the system prompt tells the LLM to stay inside the context
  const reply = await llm.invoke([
    ["system", `${SYSTEM_PROMPT}\n\nContext:\n${context}`],
    ["human", question],
  ]);

  const answer = String(reply.content).trim();
  console.log(`Answer: ${answer}`);

  // Only cite sources when the LLM actually answered
  if (!answer.startsWith(NO_ANSWER)) {
    const pages = relevant.map(([doc]) => doc.metadata["loc.pageNumber"] ?? "?");
    console.log(`Sources: page ${[...new Set(pages)].join(", ")}`);
  }
}

await ask("What is the vision of CodersGyan?");
await ask("Who won the 2018 World Cup?");
await ask("What is the CEO's monthly salary?");