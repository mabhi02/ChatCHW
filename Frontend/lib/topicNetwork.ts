import { openai } from "@ai-sdk/openai";
import { generateObject } from "ai";
import { z } from "zod";
import type { NewsArticle } from "../types/news-article";

// Define the shape of your network node and edge
export interface TopicNode {
  id: string;
  label: string;
  initialFrequency: number; // frequency for the current day
  adjustedFrequency: number; // frequency including decay from past data
  x?: number; // d3 force simulation x position
  y?: number; // d3 force simulation y position
  fx?: number | null; // d3 force simulation fixed x position
  fy?: number | null; // d3 force simulation fixed y position
}

export interface TopicEdge {
  source: string;
  target: string;
  weight: number; // co-occurrence weight
}

export interface DailyTopicNetwork {
  date: string;
  nodes: TopicNode[];
  edges: TopicEdge[];
}

// (Optional) schema for an LLM response that merges synonyms
const mergeSchema = z.object({
  mergedTopics: z.array(z.string()),
});

/**
 * Example function that calls an LLM to identify synonyms among your collected topics.
 * This is a simplistic approach: prompt the LLM with a set of topics and ask to cluster them.
 */
async function mergeTopicSynonyms(topics: string[], modelName = "gpt-4o"): Promise<string[]> {
  // Adjust prompt as needed to group/merge synonyms
  const prompt = `
We have a list of topics:
${topics.join(", ")}

Please merge topics that are synonyms or extremely similar into one single topic name (choose the best single label). Return the merged list (no duplicates).
  `;

  const { object } = await generateObject({
    model: openai(modelName),
    schema: mergeSchema,
    prompt,
  });

  return object.mergedTopics;
}

/**
 * Builds a daily topic network.
 * @param articles - array of NewsArticle for a single day
 * @param dateStr - the date (e.g. "2023-10-07") for the daily network
 * @param historicalFrequencies - pass in a dictionary of topic -> weighted frequency from the past (using decay).
 */
export async function buildDailyTopicNetwork(
  articles: NewsArticle[],
  dateStr: string,
  historicalFrequencies: Record<string, number> = {}
): Promise<DailyTopicNetwork> {
  // 1. Get all topics from today's articles
  const rawTopics: string[] = [];
  for (const article of articles) {
    rawTopics.push(...article.topics);
  }

  // 2. Deduplicate them (simple approach)
  const uniqueTopics = Array.from(new Set(rawTopics));

  // 3. Optionally call LLM to unify synonyms
  const mergedTopics = await mergeTopicSynonyms(uniqueTopics);

  // 4. Count today's frequency of each merged topic
  const todayFrequencyMap: Record<string, number> = {};
  for (const t of mergedTopics) {
    if (!todayFrequencyMap[t]) {
      todayFrequencyMap[t] = 0;
    }
  }
  // A simplistic approach: if an article's topics contain 
  // something that merges to 'Topic A', increment. 
  // In reality, you might do more advanced grouping.
  for (const article of articles) {
    for (const t of article.topics) {
      // If the merged topics are fairly direct, just see if it matches
      // In production, you'd have a better mapping from raw -> merged
      if (mergedTopics.includes(t)) {
        todayFrequencyMap[t] = (todayFrequencyMap[t] || 0) + 1;
      }
    }
  }

  // 5. Build the node list, factoring in historical frequencies
  // For a simple decay, say "decayedFreq = historicalFreq * 0.9 + currentFreq"
  const nodes: TopicNode[] = mergedTopics.map((topic) => {
    const currentFreq = todayFrequencyMap[topic] || 0;
    const historicalFreq = historicalFrequencies[topic] || 0;
    const adjusted = 0.9 * historicalFreq + currentFreq;

    return {
      id: topic,
      label: topic,
      initialFrequency: currentFreq,
      adjustedFrequency: adjusted,
    };
  });

  // 6. Build edges: if two topics co-occur in the same article, increment edge weight
  const edgeKey = (a: string, b: string) => (a < b ? `${a}||${b}` : `${b}||${a}`);
  const edgeWeights: Record<string, number> = {};

  for (const article of articles) {
    const articleTopics = article.topics.filter((t) => mergedTopics.includes(t));
    // pairwise combinations
    for (let i = 0; i < articleTopics.length; i++) {
      for (let j = i + 1; j < articleTopics.length; j++) {
        const sortedKey = edgeKey(articleTopics[i], articleTopics[j]);
        edgeWeights[sortedKey] = (edgeWeights[sortedKey] || 0) + 1;
      }
    }
  }

  const edges: TopicEdge[] = [];
  for (const [key, weight] of Object.entries(edgeWeights)) {
    const [source, target] = key.split("||");
    edges.push({ source, target, weight });
  }

  return {
    date: dateStr,
    nodes,
    edges,
  };
}

/**
 * Example function for a decay model across multiple days.
 * This function:
 * 1) Loads articles for each day,
 * 2) Builds the daily topic network,
 * 3) Recursively passes the frequencies into the next day
 */
export async function buildTopicNetworksOverTime(
  dailyArticles: Record<string, NewsArticle[]>
): Promise<DailyTopicNetwork[]> {
  let result: DailyTopicNetwork[] = [];
  let frequencyMap: Record<string, number> = {}; // store decayed frequencies

  // Sort days in ascending order
  const days = Object.keys(dailyArticles).sort();

  for (const day of days) {
    const articles = dailyArticles[day];
    const dailyNetwork = await buildDailyTopicNetwork(articles, day, frequencyMap);

    // Update the frequencyMap after building dailyNetwork
    // so that next day can use the newly decayed frequencies
    for (const node of dailyNetwork.nodes) {
      frequencyMap[node.id] = node.adjustedFrequency;
    }

    result.push(dailyNetwork);
  }

  return result;
}