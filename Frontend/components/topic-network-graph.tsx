"use client";

import * as d3 from "d3";
import { useEffect, useRef } from "react";
import type { TopicNode, TopicEdge } from "@/lib/topicNetwork";

interface TopicNetworkGraphProps {
  nodes: TopicNode[];
  edges: TopicEdge[];
  width?: number;
  height?: number;
}

export function TopicNetworkGraph({
  nodes,
  edges,
  width = 800,
  height = 600,
}: TopicNetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    // Prepare data for D3
    const d3Nodes = nodes.map((node) => ({ ...node }));
    const d3Links = edges.map((edge) => ({
      source: edge.source,
      target: edge.target,
      weight: edge.weight,
    }));

    // Clear any previous svg children
    d3.select(svgRef.current).selectAll("*").remove();

    // Setup simulation
    const simulation = d3
      .forceSimulation(d3Nodes)
      .force("link", d3.forceLink(d3Links).id((d: any) => d.id))
      .force("charge", d3.forceManyBody().strength(-80))
      .force("center", d3.forceCenter(width / 2, height / 2));

    // Draw links
    const link = d3
      .select(svgRef.current)
      .append("g")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(d3Links)
      .enter()
      .append("line")
      .attr("stroke-width", (d) => Math.sqrt(d.weight));

    // Draw nodes
    const node = d3
      .select(svgRef.current)
      .append("g")
      .attr("stroke", "white")
      .attr("stroke-width", 1.5)
      .selectAll("circle")
      .data(d3Nodes)
      .enter()
      .append("circle")
      // Use adjustedFrequency to scale radius
      .attr("r", (d: any) => 5 + Math.sqrt(d.adjustedFrequency) * 2)
      .attr("fill", "steelblue")
      .call(
        d3
          .drag<SVGCircleElement, TopicNode>()
          .on("start", dragStarted)
          .on("drag", dragged)
          .on("end", dragEnded)
      );

    const label = d3
      .select(svgRef.current)
      .append("g")
      .selectAll("text")
      .data(d3Nodes)
      .enter()
      .append("text")
      .attr("font-size", "12px")
      .attr("fill", "black")
      .text((d: any) => d.label);

    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => (d.source.x as number))
        .attr("y1", (d: any) => (d.source.y as number))
        .attr("x2", (d: any) => (d.target.x as number))
        .attr("y2", (d: any) => (d.target.y as number));

      node.attr("cx", (d: any) => d.x).attr("cy", (d: any) => d.y);

      label
        .attr("x", (d: any) => d.x + 8)
        .attr("y", (d: any) => d.y + 4);
    });

    function dragStarted(event: d3.D3DragEvent<SVGCircleElement, TopicNode, unknown>, d: TopicNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event: d3.D3DragEvent<SVGCircleElement, TopicNode, unknown>, d: TopicNode) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragEnded(event: d3.D3DragEvent<SVGCircleElement, TopicNode, unknown>, d: TopicNode) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }, [nodes, edges, width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
}