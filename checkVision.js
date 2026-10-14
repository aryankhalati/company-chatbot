import "dotenv/config";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
  apiKey: process.env.GOOGLE_API_KEY,
});

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.Index(process.env.PINECONE_INDEX);
const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex: index,
});

// Print the FULL text of the top 2 chunks, not a 150-character preview
const results = await vectorStore.similaritySearchWithScore(
  "What is the vision of CodersGyan?",
  2
);

for (const [doc, score] of results) {
  console.log(`\n=== Score ${score.toFixed(3)} ===`);
  console.log(doc.pageContent);
}