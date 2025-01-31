export type Question = {
    question: string;
    type: 'MC' | 'NUM' | 'YN' | 'MCM' | 'FREE';
    options?: { id: number | string; text: string }[];
    range?: { min: number; max: number; step: number; unit: string };
  };
  
  export type Message = {
    id: string;
    text: string;
    sender: 'user' | 'bot';
    options?: { id: string | number; text: string }[];
  };
  
  export type ScreeningAnswer = {
    question: string;
    answer: string | number | string[];
  };
  
  export type RootStackParamList = {
    QC2: undefined;
    Chat2: {
      screeningAnswers: ScreeningAnswer[];
      initialResponse: string;
      conversationId: string;
    };
  };