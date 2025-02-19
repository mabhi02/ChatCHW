'use client'

import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { User, Bot, Volume2, Globe, Mic, MicOff } from "lucide-react";
import { v4 as uuidv4 } from 'uuid';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api';
const ELEVENLABS_API_KEY = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY;

interface LanguageConfig {
  code: string;
  label: string;
  voiceId: string;
  modelId: string;
  recognitionCode: string;
  commonPhrases: Record<string, string>;
}

// Language configuration
const LANGUAGE_CONFIG: Record<string, LanguageConfig> = {
  en: {
    code: 'en',
    label: 'English',
    voiceId: '21m00Tcm4TlvDq8ikWAM',
    modelId: 'eleven_monolingual_v1',
    recognitionCode: 'en-US',
    commonPhrases: {
      'Medical Assessment': 'Medical Checkup',
      'Please specify': 'Tell us more',
      'Type your response': 'Your answer',
      'What is the patient\'s sex?': 'Are you male or female?',
      'Male': 'Male',
      'Female': 'Female',
      'Other': 'Other',
      'Other (please specify)': 'Something else (please tell us)',
    }
  },
  hi: {
    code: 'hi',
    label: 'हिंदी',
    voiceId: '29vD33N1CtxCmqQRPOHJ',
    modelId: 'eleven_multilingual_v2',
    recognitionCode: 'hi-IN',
    commonPhrases: {
      'Medical Assessment': 'स्वास्थ्य जांच',
      'Please specify': 'बताएं',
      'Type your response': 'जवाब लिखें',
      'What is the patient\'s sex?': 'आप पुरुष हैं या महिला?',
      'Male': 'पुरुष',
      'Female': 'महिला',
      'Other': 'अन्य',
      'Other (please specify)': 'कुछ और बताएं',
    }
  },
  ta: {
    code: 'ta',
    label: 'தமிழ்',
    voiceId: 'D38z5RcWu1voky8WS1ja',
    modelId: 'eleven_multilingual_v2',
    recognitionCode: 'ta-IN',
    commonPhrases: {
      'Medical Assessment': 'மருத்துவ பரிசோதனை',
      'Please specify': 'சொல்லுங்க',
      'Type your response': 'பதிலை சொல்லுங்க',
      'What is the patient\'s sex?': 'நீங்க ஆணா பெண்ணா?',
      'Male': 'ஆண்',
      'Female': 'பெண்',
      'Other': 'மற்றவை',
      'Other (please specify)': 'வேற ஏதாவது சொல்லுங்க',
    }
  }
};

type LanguageCode = keyof typeof LANGUAGE_CONFIG;

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  originalContent: string;
}

interface MessageMetadata {
  phase: 'initial' | 'followup' | 'exam' | 'complete';
  question_type?: string;
  options?: Array<{
    id: number;
    text: string;
    originalText: string;
  }>;
}

interface TranslationCache {
  [key: string]: {
    [key: string]: string;
  };
}

export function MedicalChat(): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionId] = useState(uuidv4());
  const [currentMetadata, setCurrentMetadata] = useState<MessageMetadata | null>(null);
  const [showOtherInput, setShowOtherInput] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState<LanguageCode>('en');
  const [isPlaying, setIsPlaying] = useState<string | null>(null);
  const [translationCache, setTranslationCache] = useState<TranslationCache>({});
  const [isListening, setIsListening] = useState(false);
  const [recognitionError, setRecognitionError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    startAssessment();
    initSpeechRecognition();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const translateText = async (text: string, targetLang: LanguageCode, sourceLang: LanguageCode = 'en') => {
    if (!text?.trim()) return text;
    if (targetLang === 'en') return text;

    // Check if it's a common phrase first
    const commonTranslation = LANGUAGE_CONFIG[targetLang].commonPhrases[text];
    if (commonTranslation) {
      return commonTranslation;
    }

    // Check cache
    if (translationCache[text]?.[targetLang]) {
      return translationCache[text][targetLang];
    }

    try {
      const response = await fetch(
        `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${sourceLang}|${targetLang}`
      );

      if (!response.ok) {
        throw new Error('Translation request failed');
      }

      const data = await response.json();
      
      if (data.responseStatus === 200 && data.responseData.translatedText) {
        let translatedText = data.responseData.translatedText;

        // Post-process to make more colloquial
        if (targetLang === 'ta') {
          translatedText = translatedText
            .replace(/வேண்டும்/g, 'னும்')
            .replace(/இருக்கிறது/g, 'இருக்கு')
            .replace(/செய்யுங்கள்/g, 'செய்யுங்க')
            .replace(/சொல்லுங்கள்/g, 'சொல்லுங்க')
            .replace(/வருகிறீர்கள்/g, 'வரீங்க')
            .replace(/இருக்கிறீர்கள்/g, 'இருக்கீங்க')
            .replace(/ங்கள்/g, 'ங்க')
            .replace(/கிறது/g, 'குது')
            .replace(/கிறார்/g, 'கிறார்');
        } else if (targetLang === 'hi') {
          translatedText = translatedText
            .replace(/कीजिये/g, 'करें')
            .replace(/बताइये/g, 'बताएं')
            .replace(/कहिये/g, 'कहें')
            .replace(/है।/g, 'है')
            .replace(/हैं।/g, 'हैं')
            .replace(/कीजिए/g, 'करें')
            .replace(/जाइये/g, 'जाएं');
        }
        
        setTranslationCache(prev => ({
          ...prev,
          [text]: { ...(prev[text] || {}), [targetLang]: translatedText }
        }));

        return translatedText;
      } else {
        throw new Error('Invalid translation response');
      }
    } catch (error) {
      console.error('Translation error:', error);
      return text;
    }
  };

  const initSpeechRecognition = () => {
    if (typeof window === 'undefined') return;

    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    if (!SpeechRecognition) {
      setRecognitionError('Speech recognition not supported in this browser');
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = true;

    recognitionRef.current.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((result: any) => result[0].transcript)
        .join('');
      setInputText(transcript);
    };

    recognitionRef.current.onend = () => setIsListening(false);
    recognitionRef.current.onerror = (event: any) => {
      setRecognitionError(`Recognition error: ${event.error}`);
      setIsListening(false);
    };
  };

  const toggleSpeechRecognition = () => {
    if (!recognitionRef.current) {
      initSpeechRecognition();
      if (!recognitionRef.current) return;
    }

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      try {
        recognitionRef.current.lang = LANGUAGE_CONFIG[selectedLanguage].recognitionCode;
        recognitionRef.current.start();
        setIsListening(true);
        setRecognitionError(null);
      } catch (error) {
        console.error('Failed to start speech recognition:', error);
        setRecognitionError('Failed to start speech recognition');
      }
    }
  };

  const handleSpeak = async (text: string, id: string) => {
    try {
      if (isPlaying === id) {
        audioRef.current?.pause();
        setIsPlaying(null);
        return;
      }

      if (audioRef.current) {
        audioRef.current.pause();
      }

      setIsPlaying(id);

      const selectedVoice = LANGUAGE_CONFIG[selectedLanguage];
      const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${selectedVoice.voiceId}`, {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': ELEVENLABS_API_KEY || ''
        },
        body: JSON.stringify({
          text,
          model_id: selectedVoice.modelId,
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.5
          }
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('TTS API Error:', errorText);
        throw new Error('TTS request failed');
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.onended = () => {
        setIsPlaying(null);
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
    } catch (error) {
      console.error('TTS error:', error);
      setIsPlaying(null);
    }
  };

  const handleOptionClick = async (optionId: number, optionText: string) => {
    if (isProcessing) return;
    if (optionId === 5) {
      setShowOtherInput(true);
      return;
    }

    setIsProcessing(true);
    const translatedText = selectedLanguage === 'en' 
      ? optionText 
      : await translateText(optionText, 'en', selectedLanguage);
    await handleSubmission(translatedText);
  };

  const handleSubmission = async (input: string) => {
    try {
      const translatedInput = selectedLanguage === 'en' 
        ? input 
        : await translateText(input, 'en', selectedLanguage);

      setMessages(prev => [...prev, {
        id: uuidv4(),
        type: 'user',
        content: input,
        originalContent: translatedInput
      }]);

      const response = await fetch(`${API_URL}/input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          input: translatedInput
        })
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        const translatedOutput = selectedLanguage === 'en'
          ? data.output
          : await translateText(data.output, selectedLanguage);

        setMessages(prev => [...prev, {
          id: uuidv4(),
          type: 'bot',
          content: translatedOutput,
          originalContent: data.output
        }]);

        if (data.metadata?.options) {
          const translatedOptions = await Promise.all(
            data.metadata.options.map(async (option: any) => {
              const translatedText = selectedLanguage === 'en'
                ? option.text
                : await translateText(option.text, selectedLanguage);
              
              return {
                ...option,
                text: translatedText,
                originalText: option.text
              };
            })
          );
          
          setCurrentMetadata({
            ...data.metadata,
            options: translatedOptions
          });
        } else {
          setCurrentMetadata(data.metadata);
        }
        
        setShowOtherInput(false);
      } else {
        throw new Error(data.message || 'Failed to process input');
      }

      setInputText('');
    } catch (error) {
      console.error('Error processing input:', error);
      const errorMessage = 'Error processing response. Please try again.';
      const translatedError = selectedLanguage === 'en'
        ? errorMessage
        : await translateText(errorMessage, selectedLanguage);
      
      setMessages(prev => [...prev, {
        id: uuidv4(),
        type: 'bot',
        content: translatedError,
        originalContent: errorMessage
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const startAssessment = async () => {
    try {
      setIsProcessing(true);
      const response = await fetch(`${API_URL}/start-assessment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        const translatedOutput = selectedLanguage === 'en'
          ? data.output
          : await translateText(data.output, selectedLanguage);

        setMessages([{
          id: uuidv4(),
          type: 'bot',
          content: translatedOutput,
          originalContent: data.output
        }]);

        if (data.metadata?.options) {
          const translatedOptions = await Promise.all(
            data.metadata.options.map(async (option: any) => {
              const translatedText = selectedLanguage === 'en'
                ? option.text
                : await translateText(option.text, selectedLanguage);
              
              return {
                ...option,
                text: translatedText,
                originalText: option.text
              };
            })
          );
          
          setCurrentMetadata({
            ...data.metadata,
            options: translatedOptions
          });
        } else {
          setCurrentMetadata(data.metadata);
        }
      } else {
        throw new Error(data.message || 'Failed to start assessment');
      }
    } catch (error) {
      console.error('Error starting assessment:', error);
      const errorMessage = 'Error starting assessment. Please try again.';
      const translatedError = selectedLanguage === 'en'
        ? errorMessage
        : await translateText(errorMessage, selectedLanguage);

      setMessages([{
        id: uuidv4(),
        type: 'bot',
        content: translatedError,
        originalContent: errorMessage
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleLanguageChange = async (newLang: LanguageCode) => {
    setIsProcessing(true);
    try {
      setSelectedLanguage(newLang);
      
      // Translate all existing messages
      const updatedMessages = await Promise.all(
        messages.map(async (message) => ({
          ...message,
          content: newLang === 'en'
            ? message.originalContent
            : await translateText(message.originalContent, newLang)
        }))
      );
      setMessages(updatedMessages);

      // Translate current metadata options if they exist
      if (currentMetadata?.options) {
        const translatedOptions = await Promise.all(
          currentMetadata.options.map(async (option) => ({
            ...option,
            text: newLang === 'en'
              ? option.originalText
              : await translateText(option.originalText, newLang)
          }))
        );
        
        setCurrentMetadata({
          ...currentMetadata,
          options: translatedOptions
        });
      }
    } catch (error) {
      console.error('Language change error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto h-[calc(100vh-2rem)]">
      <Card className="h-full flex flex-col">
        <CardHeader className="border-b">
          <div className="flex justify-between items-center">
            <CardTitle>
              {LANGUAGE_CONFIG[selectedLanguage].commonPhrases['Medical Assessment']}
            </CardTitle>
            <div className="flex items-center gap-2">
              <Globe className="h-4 w-4" />
              <select
                value={selectedLanguage}
                onChange={(e) => handleLanguageChange(e.target.value as LanguageCode)}
                className="h-8 rounded-md border border-input bg-background px-2 text-sm"
                disabled={isProcessing}
              >
                {Object.values(LANGUAGE_CONFIG).map(lang => (
                  <option key={lang.code} value={lang.code}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>

        <ScrollArea className="flex-1">
          <CardContent className="p-4">
            <div className="space-y-4">
              {messages.map((message) => (
                <div key={message.id}>
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
                        <div className="flex items-start gap-2">
                          <div className="whitespace-pre-wrap font-sans">
                            {message.content}
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleSpeak(message.content, message.id)}
                            disabled={isListening}
                            className={`h-6 w-6 p-0 ${isPlaying === message.id ? 'text-primary' : ''}`}
                          >
                            <Volume2 className="h-4 w-4" />
                          </Button>
                        </div>
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
          {currentMetadata?.options && !showOtherInput && (
            <div className="grid grid-cols-1 gap-2 mb-4">
              {currentMetadata.options.map((option) => (
                <div key={option.id} className="flex items-center gap-2">
                  <Button
                    onClick={() => handleOptionClick(option.id, option.text)}
                    disabled={isProcessing}
                    variant={option.id === 5 ? "secondary" : "default"}
                    className="flex-grow justify-start text-left hover:bg-primary/90"
                  >
                    {option.text}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleSpeak(option.text, `option-${option.id}`)}
                    disabled={isListening}
                    className={`h-8 w-8 p-0 ${isPlaying === `option-${option.id}` ? 'text-primary' : ''}`}
                  >
                    <Volume2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          {(showOtherInput || currentMetadata?.question_type === 'NUM' || currentMetadata?.question_type === 'FREE') && (
            <form onSubmit={(e) => {
              e.preventDefault();
              if (!inputText.trim() || isProcessing) return;
              handleSubmission(inputText);
            }} className="flex space-x-2">
              <Input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={showOtherInput 
                  ? LANGUAGE_CONFIG[selectedLanguage].commonPhrases['Please specify']
                  : LANGUAGE_CONFIG[selectedLanguage].commonPhrases['Type your response']}
                className="flex-1"
                disabled={isProcessing || isListening}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={toggleSpeechRecognition}
                disabled={isProcessing || !!recognitionError}
                className={`h-10 w-10 p-0 ${isListening ? 'text-primary animate-pulse' : ''}`}
              >
                {isListening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
              </Button>
              <Button 
                type="submit" 
                disabled={isProcessing || (!inputText.trim() && !isListening)}
              >
                Send
              </Button>
            </form>
          )}
          
          {recognitionError && (
            <div className="mt-2 text-sm text-red-500">
              {recognitionError}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

export default MedicalChat;