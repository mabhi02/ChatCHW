'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { User, Bot } from "lucide-react";

interface Option {
  id: number;
  text: string;
}

interface Range {
  min: number;
  max: number;
  step: number;
  unit: string;
}

interface Question {
  question: string;
  type: 'MC' | 'NUM' | 'MCM' | 'FREE';
  options?: Option[];
  range?: Range;
}

interface Message {
  type: 'user' | 'bot';
  content: string;
  options?: Option[];
}

const initialQuestions: Question[] = [
  {
    question: "What is the patient's sex?",
    type: "MC",
    options: [
      {id: 1, text: "Male"},
      {id: 2, text: "Female"},
      {id: 3, text: "Non-binary"},
      {id: 4, text: "Other"},
      {id: 5, text: "Other (please specify)"}
    ]
  },
  {
    question: "What is the patient's age?",
    type: "NUM",
    range: {
      min: 0,
      max: 120,
      step: 1,
      unit: "years"
    }
  },
  {
    question: "Does the patient have a caregiver?",
    type: "MC",
    options: [
      {id: 1, text: "Yes"},
      {id: 2, text: "No"},
      {id: 3, text: "Not sure"},
      {id: 4, text: "Sometimes"},
      {id: 5, text: "Other (please specify)"}
    ]
  },
  {
    question: "Who is accompanying the patient?",
    type: "MCM",
    options: [
      {id: 1, text: "None"},
      {id: 2, text: "Relatives"},
      {id: 3, text: "Friends"},
      {id: 4, text: "Health workers"},
      {id: 5, text: "Other (please specify)"}
    ]
  },
  {
    question: "Please describe what brings you here today",
    type: "FREE"
  }
];

const followUpQuestions = [
  {
    question: "How long have you been experiencing these symptoms?",
    options: [
      {id: 1, text: "Less than 24 hours"},
      {id: 2, text: "1-3 days"},
      {id: 3, text: "4-7 days"},
      {id: 4, text: "More than a week"},
      {id: 5, text: "Other (please specify)"}
    ]
  },
  {
    question: "Rate your pain level:",
    options: [
      {id: 1, text: "No pain"},
      {id: 2, text: "Mild"},
      {id: 3, text: "Moderate"},
      {id: 4, text: "Severe"},
      {id: 5, text: "Other (please specify)"}
    ]
  },
  {
    question: "Have you taken any medication for this?",
    options: [
      {id: 1, text: "Yes, prescribed"},
      {id: 2, text: "Yes, over the counter"},
      {id: 3, text: "No"},
      {id: 4, text: "Not sure"},
      {id: 5, text: "Other (please specify)"}
    ]
  }
];

export function MedicalChat(): JSX.Element {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [answers, setAnswers] = useState<Record<number, any>>({});
  const [showChat, setShowChat] = useState<boolean>(false);
  const [otherText, setOtherText] = useState<string>("");
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState<number>(0);
  const [showCustomInput, setShowCustomInput] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleQuestionAnswer = (answer: any, questionIndex: number): void => {
    setAnswers(prev => ({
      ...prev,
      [questionIndex]: answer
    }));
    
    if (questionIndex < initialQuestions.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      setShowChat(true);
      setMessages([{
        type: 'bot',
        content: followUpQuestions[0].question,
        options: followUpQuestions[0].options
      }]);
    }
  };

  const handleChatResponse = (option: Option) => {
    setMessages(prev => [...prev, {
      type: 'user',
      content: option.text
    }]);

    if (option.text === "Other (please specify)") {
      setShowCustomInput(true);
      return;
    }

    if (currentQuestion < followUpQuestions.length - 1) {
      const nextQuestion = followUpQuestions[currentQuestion + 1];
      setCurrentQuestion(currentQuestion + 1);
      setTimeout(() => {
        setMessages(prev => [...prev, {
          type: 'bot',
          content: nextQuestion.question,
          options: nextQuestion.options
        }]);
      }, 500);
    }
  };

  const handleCustomInput = (text: string) => {
    setMessages(prev => [...prev, {
      type: 'user',
      content: text
    }]);
    setShowCustomInput(false);
    setOtherText("");

    if (currentQuestion < followUpQuestions.length - 1) {
      const nextQuestion = followUpQuestions[currentQuestion + 1];
      setCurrentQuestion(currentQuestion + 1);
      setTimeout(() => {
        setMessages(prev => [...prev, {
          type: 'bot',
          content: nextQuestion.question,
          options: nextQuestion.options
        }]);
      }, 500);
    }
  };

  const renderQuestion = (): JSX.Element => {
    const currentQuestion = initialQuestions[currentStep];

    return (
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>{currentQuestion.question}</CardTitle>
        </CardHeader>
        <CardContent>
          {currentQuestion.type === "MC" && currentQuestion.options && (
            <div className="grid grid-cols-1 gap-2">
              {currentQuestion.options.map((option) => (
                <Button
                  key={option.id}
                  onClick={() => handleQuestionAnswer(option.text, currentStep)}
                  className="w-full justify-start text-left"
                  variant="outline"
                >
                  {option.text}
                </Button>
              ))}
            </div>
          )}

          {currentQuestion.type === "NUM" && currentQuestion.range && (
            <div className="space-y-4">
              <Input
                type="number"
                min={currentQuestion.range.min}
                max={currentQuestion.range.max}
                step={currentQuestion.range.step}
                value={otherText}
                onChange={(e) => setOtherText(e.target.value)}
                className="w-full"
              />
              <Button
                onClick={() => {
                  const value = parseInt(otherText);
                  if (!isNaN(value) && value >= 0 && value <= 120) {
                    handleQuestionAnswer(value, currentStep);
                    setOtherText("");
                  }
                }}
                className="w-full"
              >
                Continue
              </Button>
            </div>
          )}

          {currentQuestion.type === "MCM" && currentQuestion.options && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 gap-2">
                {currentQuestion.options.map((option) => (
                  <Button
                    key={option.id}
                    onClick={() => {
                      const newSelected = selectedOptions.includes(option.text)
                        ? selectedOptions.filter(item => item !== option.text)
                        : [...selectedOptions, option.text];
                      setSelectedOptions(newSelected);
                    }}
                    className={`w-full justify-start text-left ${
                      selectedOptions.includes(option.text) ? "bg-primary text-primary-foreground" : ""
                    }`}
                    variant="outline"
                  >
                    {option.text}
                  </Button>
                ))}
              </div>
              <Button
                onClick={() => handleQuestionAnswer(selectedOptions, currentStep)}
                className="w-full"
              >
                Continue
              </Button>
            </div>
          )}

          {currentQuestion.type === "FREE" && (
            <div className="space-y-4">
              <Input
                type="text"
                placeholder="Type your answer here..."
                value={otherText}
                onChange={(e) => setOtherText(e.target.value)}
                className="w-full"
              />
              <Button
                onClick={() => handleQuestionAnswer(otherText, currentStep)}
                className="w-full"
              >
                Continue
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };
  

  const renderChat = (): JSX.Element => {
    return (
      <div className="flex flex-col h-[calc(100vh-14rem)]">
        <div className="mb-4">
          <Alert>
            <AlertDescription>
              <div className="font-semibold mb-1">Patient Information</div>
              {Object.entries(answers).map(([key, value]) => (
                <div key={key} className="text-sm">
                  {initialQuestions[parseInt(key)].question}: {Array.isArray(value) ? value.join(", ") : value}
                </div>
              ))}
            </AlertDescription>
          </Alert>
        </div>

        <Card className="flex-1 flex flex-col min-h-0">
          <CardHeader className="border-b">
            <CardTitle>Medical Consultation</CardTitle>
          </CardHeader>
          <ScrollArea className="flex-1">
            <CardContent className="p-4">
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div key={index}>
                    <div
                      className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`flex items-start space-x-2 max-w-[80%] ${
                          message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                        }`}
                      >
                        <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                          {message.type === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                        </div>
                        <div
                          className={`rounded-lg p-3 ${
                            message.type === 'user'
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted'
                          }`}
                        >
                          {message.content}
                        </div>
                      </div>
                    </div>
                    {message.type === 'bot' && message.options && (
                      <div className="ml-12 mt-2 space-y-2">
                        {message.options.map((option) => (
                          <Button
                            key={option.id}
                            onClick={() => handleChatResponse(option)}
                            className="w-full justify-start text-left"
                            variant="outline"
                          >
                            {option.text}
                          </Button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            </CardContent>
          </ScrollArea>
          <div className="border-t p-4">
            {showCustomInput && (
              <div className="flex space-x-2">
                <Input
                  value={otherText}
                  onChange={(e) => setOtherText(e.target.value)}
                  placeholder="Type your response..."
                  className="flex-1"
                />
                <Button onClick={() => handleCustomInput(otherText)}>Send</Button>
              </div>
            )}
          </div>
        </Card>
      </div>
    );
  };

  return (
    <div className="w-full">
      {!showChat ? renderQuestion() : renderChat()}
    </div>
  );
}

export default MedicalChat;