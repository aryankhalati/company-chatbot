import 'dotenv/config';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

export async function indexTheDocument(filePath) {
  // 1. Load: one document per page
  const loader = new PDFLoader(filePath);
  const docs = await loader.load();

  // 2. Chunk: cut pages into smaller pieces
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const chunks = await splitter.splitDocuments(docs);
  console.log('Pages:', docs.length, '-> Chunks:', chunks.length);

  // 3 + 4. Embed each chunk and store the vectors in Pinecone
  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: 'gemini-embedding-001',
  });

  const pinecone = new Pinecone(); // reads PINECONE_API_KEY from .env
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

  const vectorStore = await PineconeStore.fromDocuments(chunks, embeddings, {
    pineconeIndex,
    maxConcurrency: 5, // upload a few batches at a time
  });

  console.log('Stored', chunks.length, 'chunks in Pinecone');
  return vectorStore;
}