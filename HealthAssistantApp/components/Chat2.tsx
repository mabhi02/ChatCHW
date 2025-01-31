import React, { useState, useEffect, useRef } from 'react';
import { 
  View, 
  Text, 
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
  TextInput,
  Platform,
  KeyboardAvoidingView
} from 'react-native';
import { RouteProp } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';

const API_URL = 'http://localhost:5000';
const { height: WINDOW_HEIGHT } = Dimensions.get('window');

// Debug logging function
const debug = (message: string, data?: any) => {
  console.log(`[Chat2 Debug] ${message}`, data || '');
};

type ScreeningAnswer = {
  question: string;
  answer: string | number | string[];
};

type RootStackParamList = {
  QC2: undefined;
  Chat2: {
    screeningAnswers: ScreeningAnswer[];
    initialResponse: string;
    conversationId: string;
  };
};

type Chat2Props = {
  route: RouteProp<RootStackParamList, 'Chat2'>;
  navigation: StackNavigationProp<RootStackParamList>;
};

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  options?: Array<{
    id: number | string;
    text: string;
  }>;
  requiresTextInput?: boolean;
};

const Chat2: React.FC<Chat2Props> = ({ route, navigation }) => {
  const { screeningAnswers, initialResponse, conversationId } = route.params;
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [currentQuestion, setCurrentQuestion] = useState<Message | null>(null);
  const scrollViewRef = useRef<ScrollView>(null);

  useEffect(() => {
    debug('Component mounted');
    displayScreeningAnswers();
    handleInitialSymptom();
  }, []);

  const handleInitialSymptom = () => {
    debug('Handling initial symptom');
    const initialSymptom = screeningAnswers[screeningAnswers.length - 1].answer;
    if (typeof initialSymptom === 'string') {
      handleSendMessage(initialSymptom, true);
    }
  };

  const displayScreeningAnswers = () => {
    debug('Displaying screening answers');
    if (screeningAnswers?.length > 0) {
      const summaryText = screeningAnswers
        .map(answer => `${answer.question}: ${answer.answer}`)
        .join('\n');
      addBotMessage(summaryText);
    }
  };

  const shouldShowTextInput = (question: string) => {
    // Only require text input for very specific cases that absolutely need detailed description
    const textInputTriggers = [
      'describe the blood',
      'describe your symptoms',
      'describe the pain',
      'tell me more about',
      'additional details'
    ];
    return textInputTriggers.some(trigger => 
      question.toLowerCase().includes(trigger)
    );
  };

  const addBotMessage = (text: string, options?: Array<{ id: number | string; text: string; }>) => {
    debug('Adding bot message:', text);
    const requiresTextInput = shouldShowTextInput(text);
    debug('Requires text input:', requiresTextInput);
    
    // Remove default options - only use the ones provided by the backend
    const newMessage: Message = {
      id: Math.random().toString(),
      text,
      sender: 'bot',
      options: requiresTextInput ? undefined : options,
      requiresTextInput
    };
    
    setMessages(prev => [...prev, newMessage]);
    setCurrentQuestion(newMessage);
    scrollToBottom();
  };

  const handleSendMessage = async (text: string, isInitial: boolean = false) => {
    debug('Sending message:', { text, isInitial });
    const userMessage: Message = {
      id: Math.random().toString(),
      text,
      sender: 'user'
    };
    setMessages(prev => [...prev, userMessage]);
    setTextInput('');
    scrollToBottom();

    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          screeningAnswers,
          conversationHistory: messages.map(m => ({
            role: m.sender,
            content: m.text
          })),
          conversationId
        }),
      });

      const data = await response.json();
      debug('Received response:', data);
      addBotMessage(data.question, data.options);
    } catch (error) {
      debug('Error:', error);
      addBotMessage("Sorry, something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleOptionPress = (option: { id: number | string; text: string }) => {
    debug('Option pressed:', option);
    handleSendMessage(option.text);
  };

  const handleTextInputSubmit = () => {
    debug('Text input submitted:', textInput);
    if (textInput.trim()) {
      handleSendMessage(textInput.trim());
    }
  };

  const scrollToBottom = () => {
    debug('Scrolling to bottom');
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  };

  const renderMessage = (message: Message) => (
    <View 
      key={message.id}
      style={[
        styles.messageContainer,
        message.sender === 'user' ? styles.userMessage : styles.botMessage
      ]}
    >
      <Text style={[
        styles.messageText,
        message.sender === 'user' ? styles.userText : styles.botText
      ]}>
        {message.text}
      </Text>

      {message.options && (
        <View style={styles.optionsContainer}>
          {message.options.map(option => (
            <TouchableOpacity
              key={option.id}
              style={styles.optionButton}
              onPress={() => handleOptionPress(option)}
            >
              <Text style={styles.optionText}>{option.text}</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}
    </View>
  );

  return (
    <KeyboardAvoidingView 
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={styles.mainContainer}
      keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}
    >
      <View style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity 
            onPress={() => navigation.goBack()}
            style={styles.headerButton}
          >
            <Text style={styles.headerButtonText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Health Assistant</Text>
          <View style={styles.headerButton} />
        </View>

        <View style={styles.contentContainer}>
          <ScrollView
            ref={scrollViewRef}
            style={styles.scrollView}
            contentContainerStyle={styles.scrollContent}
            showsVerticalScrollIndicator={true}
            onContentSizeChange={scrollToBottom}
          >
            {messages.map(renderMessage)}
            {loading && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator color="#007AFF" size="small" />
              </View>
            )}
          </ScrollView>

          {currentQuestion?.requiresTextInput && (
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.textInput}
                value={textInput}
                onChangeText={setTextInput}
                placeholder="Type your response..."
                placeholderTextColor="#666"
                returnKeyType="send"
                onSubmitEditing={handleTextInputSubmit}
              />
              <TouchableOpacity
                style={[styles.sendButton, !textInput.trim() && styles.buttonDisabled]}
                onPress={handleTextInputSubmit}
                disabled={!textInput.trim()}
              >
                <Text style={styles.sendButtonText}>Send</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  mainContainer: {
    flex: 1,
  },
  container: {
    flex: 1,
    backgroundColor: '#121212',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    backgroundColor: '#1E1E1E',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  headerButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerButtonText: {
    color: '#007AFF',
    fontSize: 24,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  contentContainer: {
    flex: 1,
    maxHeight: WINDOW_HEIGHT - 90, // Reduced padding, just account for header
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 16,
  },
  messageContainer: {
    maxWidth: '80%',
    marginBottom: 16,
    padding: 16,
    borderRadius: 16,
    borderWidth: 1,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007AFF',
    borderColor: '#0056b3',
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#1E1E1E',
    borderColor: '#333',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  userText: {
    color: '#fff',
  },
  botText: {
    color: '#fff',
  },
  optionsContainer: {
    marginTop: 12,
    gap: 8,
  },
  optionButton: {
    backgroundColor: '#2C2C2E',
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  optionText: {
    color: '#007AFF',
    fontSize: 15,
  },
  loadingContainer: {
    padding: 16,
    alignItems: 'center',
  },
  inputContainer: {
    padding: 8,
    backgroundColor: '#1E1E1E',
    borderTopWidth: 1,
    borderTopColor: '#333',
    flexDirection: 'row',
    gap: 8,
    alignItems: 'center',
  },
  textInput: {
    flex: 1,
    backgroundColor: '#2C2C2E',
    borderRadius: 12,
    padding: 12,
    color: '#fff',
    fontSize: 16,
    minHeight: 40,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    borderRadius: 12,
    paddingHorizontal: 16,
    justifyContent: 'center',
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
});

export default Chat2;