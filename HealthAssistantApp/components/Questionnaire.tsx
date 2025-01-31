import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  TextInput, 
  Pressable, 
  StyleSheet,
  Animated,
  Keyboard,
  Dimensions,
  ScrollView,
  Alert,
} from 'react-native';
import { RootStackParamList } from '../App';
import type { StackNavigationProp } from '@react-navigation/stack';

const API_URL = 'http://localhost:5000';
const { width } = Dimensions.get('window');

type QuestionType = 'MC' | 'NUM' | 'YN' | 'MCM' | 'FREE';

type Question = {
  question: string;
  type: QuestionType;
  options?: Array<{
    id: number | string;
    text: string;
  }>;
  range?: {
    min: number;
    max: number;
    step: number;
    unit: string;
  };
};

type Props = {
  navigation: StackNavigationProp<RootStackParamList, 'QC2'>;
};

const QC2: React.FC<Props> = ({ navigation }) => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [responses, setResponses] = useState<(string | number | string[])[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [textInputValue, setTextInputValue] = useState('');
  
  const fadeAnim = new Animated.Value(0);
  const slideAnim = new Animated.Value(width);
  const progressWidth = new Animated.Value(0);

  const fetchQuestions = async () => {
    try {
      const response = await fetch(`${API_URL}/initial_questions`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setQuestions(data);
      setLoading(false);
    } catch (err) {
      setError('Failed to load questions');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQuestions();
  }, []);

  useEffect(() => {
    if (questions.length === 0) return;
    
    const progress = (currentQuestion + 1) / questions.length;
    fadeAnim.setValue(0);
    slideAnim.setValue(width);
    
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.spring(slideAnim, {
        toValue: 0,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }),
      Animated.timing(progressWidth, {
        toValue: progress,
        duration: 600,
        useNativeDriver: false,
      })
    ]).start();

    // Reset text input when question changes
    setTextInputValue(responses[currentQuestion]?.toString() || '');
  }, [currentQuestion, questions.length]);

  const handleOptionSelect = (value: string | number | string[]) => {
    const newResponses = [...responses];
    newResponses[currentQuestion] = value;
    setResponses(newResponses);

    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: -width,
        duration: 300,
        useNativeDriver: true,
      })
    ]).start(() => {
      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(prev => prev + 1);
      } else {
        handleSubmit();
      }
    });
  };

  const handleMCMSelect = (optionId: string | number) => {
    const currentValue = responses[currentQuestion] as string[] || [];
    const newValue = currentValue.includes(String(optionId))
      ? currentValue.filter(id => id !== String(optionId))
      : [...currentValue, String(optionId)];
    const newResponses = [...responses];
    newResponses[currentQuestion] = newValue;
    setResponses(newResponses);
  };

  const handleTextInput = (text: string) => {
    setTextInputValue(text);
    const newResponses = [...responses];
    newResponses[currentQuestion] = text;
    setResponses(newResponses);
  };

  const handleNumberInput = (text: string) => {
    const num = text === '' ? '' : Number(text);
    const question = questions[currentQuestion];
    if (question.range && typeof num === 'number') {
      if (num < question.range.min || num > question.range.max) {
        Alert.alert('Invalid Input', `Please enter a number between ${question.range.min} and ${question.range.max}`);
        return;
      }
    }
    handleTextInput(text);
  };

  const handleSubmit = async () => {
    Keyboard.dismiss();
    const screeningAnswers = questions.map((q, index) => ({
      question: q.question,
      answer: responses[index]
    }));

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: responses[responses.length - 1],
          screeningAnswers,
          conversationHistory: []
        }),
      });

      const data = await response.json();
      
      Animated.sequence([
        Animated.timing(progressWidth, {
          toValue: 1,
          duration: 300,
          useNativeDriver: false,
        }),
        Animated.delay(500)
      ]).start(() => {
        navigation.navigate('Chat2', {
          screeningAnswers,
          initialResponse: data.response,
          conversationId: data.conversationId
        });
      });
    } catch (err) {
      setError('Failed to start chat');
    }
  };

  const renderQuestionInput = () => {
    if (questions.length === 0) return null;
    const question = questions[currentQuestion];

    switch (question.type) {
      case 'FREE':
        return (
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              placeholder="Type your answer here..."
              placeholderTextColor="#666"
              onChangeText={handleTextInput}
              value={textInputValue}
              multiline
              numberOfLines={4}
              textAlignVertical="top"
            />
            <Pressable 
              style={({pressed}) => [
                styles.nextButton,
                pressed && styles.buttonPressed,
                !textInputValue && styles.buttonDisabled
              ]}
              onPress={() => textInputValue ? handleOptionSelect(textInputValue) : null}
              disabled={!textInputValue}
            >
              <Text style={styles.nextButtonText}>Next</Text>
            </Pressable>
          </View>
        );

      case 'NUM':
        return (
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              keyboardType="numeric"
              placeholder={`Enter a number (${question.range?.min}-${question.range?.max})`}
              placeholderTextColor="#666"
              onChangeText={handleNumberInput}
              value={textInputValue}
            />
            <Pressable 
              style={({pressed}) => [
                styles.nextButton,
                pressed && styles.buttonPressed,
                !textInputValue && styles.buttonDisabled
              ]}
              onPress={() => textInputValue ? handleOptionSelect(Number(textInputValue)) : null}
              disabled={!textInputValue}
            >
              <Text style={styles.nextButtonText}>Next</Text>
            </Pressable>
          </View>
        );

      case 'MCM':
        return (
          <ScrollView style={styles.optionsContainer}>
            {question.options?.map(option => {
              const isSelected = (responses[currentQuestion] as string[] || []).includes(String(option.id));
              return (
                <Pressable
                  key={option.id}
                  style={({pressed}) => [
                    styles.optionButton,
                    isSelected && styles.selectedOption,
                    pressed && styles.buttonPressed
                  ]}
                  onPress={() => handleMCMSelect(option.id)}
                >
                  <Text style={[
                    styles.optionText,
                    isSelected && styles.selectedOptionText
                  ]}>{option.text}</Text>
                </Pressable>
              );
            })}
            <Pressable 
              style={({pressed}) => [
                styles.nextButton,
                pressed && styles.buttonPressed,
                !(responses[currentQuestion] as string[] || []).length && styles.buttonDisabled
              ]}
              onPress={() => responses[currentQuestion] ? handleOptionSelect(responses[currentQuestion]) : null}
              disabled={!(responses[currentQuestion] as string[] || []).length}
            >
              <Text style={styles.nextButtonText}>Next</Text>
            </Pressable>
          </ScrollView>
        );

      default: // MC and YN
        return (
          <ScrollView style={styles.optionsContainer}>
            {question.options?.map(option => (
              <Pressable
                key={option.id}
                style={({pressed}) => [
                  styles.optionButton,
                  pressed && styles.buttonPressed
                ]}
                onPress={() => handleOptionSelect(option.id)}
              >
                <Text style={styles.optionText}>{option.text}</Text>
              </Pressable>
            ))}
          </ScrollView>
        );
    }
  };

  const progressBarWidth = progressWidth.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Health Assessment</Text>
        <View style={styles.progressBarContainer}>
          <Animated.View 
            style={[
              styles.progressBar,
              { width: progressBarWidth }
            ]} 
          />
        </View>
      </View>

      {loading ? (
        <View style={styles.centerContainer}>
          <Text style={styles.loadingText}>Loading questions...</Text>
        </View>
      ) : error ? (
        <View style={styles.centerContainer}>
          <Text style={styles.errorText}>{error}</Text>
          <Pressable 
            style={({pressed}) => [
              styles.retryButton,
              pressed && styles.buttonPressed
            ]}
            onPress={fetchQuestions}
          >
            <Text style={styles.retryText}>Retry</Text>
          </Pressable>
        </View>
      ) : (
        <ScrollView style={styles.contentContainer}>
          <Animated.View 
            style={[
              styles.questionContainer,
              {
                opacity: fadeAnim,
                transform: [{ translateX: slideAnim }]
              }
            ]}
          >
            <Text style={styles.questionCount}>
              Question {currentQuestion + 1} of {questions.length}
            </Text>
            
            <Text style={styles.questionText}>{questions[currentQuestion]?.question}</Text>
            
            {renderQuestionInput()}
          </Animated.View>
        </ScrollView>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
  },
  contentContainer: {
    flex: 1,
  },
  header: {
    padding: 16,
    backgroundColor: '#1E1E1E',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 16,
  },
  progressBarContainer: {
    height: 4,
    backgroundColor: '#333',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#007AFF',
  },
  questionContainer: {
    padding: 20,
    minHeight: 400,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  questionCount: {
    fontSize: 14,
    color: '#888',
    marginBottom: 8,
  },
  questionText: {
    fontSize: 24,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 32,
  },
  optionsContainer: {
    maxHeight: 400,
  },
  optionButton: {
    backgroundColor: '#1E1E1E',
    padding: 20,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
    marginBottom: 12,
  },
  selectedOption: {
    backgroundColor: '#1E88E5',
    borderColor: '#1565C0',
  },
  buttonPressed: {
    opacity: 0.7,
    transform: [{ scale: 0.98 }],
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  optionText: {
    fontSize: 16,
    color: '#fff',
  },
  selectedOptionText: {
    color: '#fff',
    fontWeight: '600',
  },
  inputContainer: {
    gap: 16,
  },
  input: {
    backgroundColor: '#1E1E1E',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 12,
    padding: 16,
    color: '#fff',
    fontSize: 16,
    minHeight: 100,
  },
  nextButton: {
    backgroundColor: '#007AFF',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  nextButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingText: {
    color: '#888',
    fontSize: 16,
  },
  errorText: {
    color: '#ff4444',
    fontSize: 16,
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    minWidth: 120,
    alignItems: 'center',
  },
  retryText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
});

export default QC2;