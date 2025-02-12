'use client'

import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { User, Bot } from "lucide-react";
import { v4 as uuidv4 } from 'uuid';


const API_URL = 'http://localhost:5000/api';

interface Message {
  type: 'user' | 'bot';
  content: string;
}

interface MessageMetadata {
  phase: 'initial' | 'followup' | 'exam' | 'complete';
  question_type?: string;
  options?: Array<{
    id: number;
    text: string;
  }>;
}

export function MedicalChat(): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionId] = useState(uuidv4());
  const [currentMetadata, setCurrentMetadata] = useState<MessageMetadata | null>(null);
  const [showOtherInput, setShowOtherInput] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    startAssessment();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleOptionClick = async (optionId: number, optionText: string) => {
    if (isProcessing) return;

    if (optionId === 5) {
      setShowOtherInput(true);
      return;
    }

    setIsProcessing(true);
    // For exam phase, send the ID instead of the text
    const inputToSend = currentMetadata?.phase === 'exam' ? optionId.toString() : optionText;
    await handleSubmission(inputToSend);
  };

  const cleanBotMessage = (message: string) => {
    // Debug log
    console.log('Current metadata:', currentMetadata);
    console.log('Message:', message);
  
    // For examinations - check both metadata and message content
    if (message.includes('Recommended Examination:') || 
        currentMetadata?.phase === 'exam' ||
        currentMetadata?.question_type === 'EXAM') {
      console.log('In examination phase');
      return message;  // Return full examination text
    }
  
    // For initial questions and followups only: show just the question
    if (currentMetadata?.phase !== 'complete' && currentMetadata?.options) {
      return message.split('\n')[0].trim();
    }
  
    // For everything else, show full message
    return message;
  };
  
  const renderMessage = (message: Message) => {
    // More debug logs
    console.log('Rendering message:', message);
    console.log('Current phase:', currentMetadata?.phase);
  
    return (
      <div className="whitespace-pre-wrap font-sans">
        {message.content}
      </div>
    );
  };

  
  const handleSubmission = async (input: string) => {
    try {
      // Add user's response to chat
      setMessages(prev => [...prev, {
        type: 'user',
        content: input
      }]);

      // Send input to backend
      const response = await fetch(`${API_URL}/input`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: sessionId,
          input: input
        })
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        setMessages(prev => [...prev, {
          type: 'bot',
          content: cleanBotMessage(data.output)
        }]);
        setCurrentMetadata(data.metadata);
        setShowOtherInput(false);
      } else {
        throw new Error(data.message || 'Failed to process input');
      }

      setInputText('');
    } catch (err) {
      console.error('Error processing input:', err);
      setMessages(prev => [...prev, {
        type: 'bot',
        content: 'Error processing response. Please try again.'
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleInputSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isProcessing) return;

    setIsProcessing(true);
    await handleSubmission(inputText);
  };

  const startAssessment = async () => {
    try {
      setIsProcessing(true);
      const response = await fetch(`${API_URL}/start-assessment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: sessionId
        })
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        setMessages([{
          type: 'bot',
          content: cleanBotMessage(data.output)
        }]);
        setCurrentMetadata(data.metadata);
      } else {
        throw new Error(data.message || 'Failed to start assessment');
      }
    } catch (err) {
      console.error('Error starting assessment:', err);
      setMessages([{
        type: 'bot',
        content: 'Error starting assessment. Please try again.'
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const shouldShowTextInput = () => {
    if (!currentMetadata) return false;
    if (currentMetadata.phase === 'complete') return false;
    if (showOtherInput) return true;
    return currentMetadata.question_type === 'NUM' || 
           currentMetadata.question_type === 'FREE';
  };

  return (
    <div className="w-full max-w-4xl mx-auto h-[calc(100vh-2rem)]">
      <Card className="h-full flex flex-col">
        <CardHeader className="border-b">
          <CardTitle>Medical Assessment</CardTitle>
        </CardHeader>
        <ScrollArea className="flex-1">
          <CardContent className="p-4">
            <div className="space-y-4">
            {messages.map((message, index) => (
              <div key={index}>
                <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`flex items-start space-x-2 max-w-[80%] ${
                    message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                  }`}>
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                      {message.type === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                    </div>
                    <div className={`rounded-lg p-3 ${
                      message.type === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted'
                    }`}>
                      {renderMessage(message)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
              <div ref={messagesEndRef} />
            </div>
          </CardContent>
        </ScrollArea>
        <div className="border-t p-4">
          {currentMetadata?.options && !shouldShowTextInput() && (
            <div className="grid grid-cols-1 gap-2 mb-4">
              {currentMetadata.options.map((option) => (
                <Button
                  key={option.id}
                  onClick={() => handleOptionClick(option.id, option.text)}
                  disabled={isProcessing}
                  variant={option.id === 5 ? "secondary" : "default"}
                  className={`w-full justify-start text-left hover:bg-primary/90 ${
                    option.id === 5 ? 'bg-background text-foreground hover:bg-accent' : ''
                  }`}
                >
                  {option.text}
                </Button>
              ))}
            </div>
          )}
          {shouldShowTextInput() && (
            <form onSubmit={handleInputSubmit} className="flex space-x-2">
              <Input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={showOtherInput ? "Please specify..." : "Type your response..."}
                className="flex-1"
                disabled={isProcessing}
              />
              <Button 
                type="submit" 
                disabled={isProcessing}
              >
                Send
              </Button>
            </form>
          )}
        </div>
      </Card>
    </div>
  );
}

export default MedicalChat;