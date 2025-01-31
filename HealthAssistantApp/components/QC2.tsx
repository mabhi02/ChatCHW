import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  TextInput, 
  Pressable, 
  StyleSheet,
  ScrollView,
  ActivityIndicator
} from 'react-native';
import type { StackNavigationProp } from '@react-navigation/stack';

// Define the types for the navigation stack
type RootStackParamList = {
  QC2: undefined;
  Chat2: {
    screeningAnswers: ScreeningAnswer[];
    initialResponse: string;
    conversationId: string;
  };
};

// Define types for questions and answers
type Question = {
  question: string;
  type: 'MC' | 'NUM' | 'YN' | 'MCM' | 'FREE';
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

type ScreeningAnswer = {
  question: string;
  answer: string | number | string[];
};

type Props = {
  navigation: StackNavigationProp<RootStackParamList, 'QC2'>;
};

const API_URL = 'http://localhost:5000';

const QC2: React.FC<Props> = ({ navigation }) => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [responses, setResponses] = useState<(string | number | string[])[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchQuestions();
  }, []);

  const fetchQuestions = async () => {
    try {
      const response = await fetch(`${API_URL}/initial_questions`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setQuestions(data);
      setResponses(new Array(data.length).fill(undefined));
      setLoading(false);
    } catch (err) {
      console.error('Error fetching questions:', err);
      setError('Failed to load questions. Please try again.');
      setLoading(false);
    }
  };

  const handleResponse = (value: string | number | string[]) => {
    setResponses(prev => {
      const newResponses = [...prev];
      newResponses[currentQuestion] = value;
      return newResponses;
    });
  };

  const handleNext = async () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(prev => prev + 1);
    } else {
      try {
        const screeningAnswers: ScreeningAnswer[] = questions.map((q, index) => ({
          question: q.question,
          answer: responses[index] ?? ''
        }));

        const chatResponse = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: responses[responses.length - 1],
            screeningAnswers,
            conversationHistory: []
          }),
        });

        if (!chatResponse.ok) throw new Error('Failed to start chat');
        const data: { response: string; conversationId: string } = await chatResponse.json();
        
        navigation.navigate('Chat2', {
          screeningAnswers,
          initialResponse: data.response,
          conversationId: data.conversationId
        });
      } catch (err) {
        console.error('Error starting chat:', err);
        setError('Failed to start chat. Please try again.');
      }
    }
  };

  const renderQuestion = () => {
    const question = questions[currentQuestion];
    if (!question) return null;

    switch (question.type) {
      case 'FREE':
        return (
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.textInput}
              placeholder="Type your answer here..."
              placeholderTextColor="#666"
              value={responses[currentQuestion]?.toString() ?? ''}
              onChangeText={(text) => handleResponse(text)}
              multiline
              numberOfLines={4}
              textAlignVertical="top"
            />
          </View>
        );

      case 'NUM':
        return (
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.numberInput}
              keyboardType="numeric"
              placeholder={`Enter a number (${question.range?.min}-${question.range?.max})`}
              placeholderTextColor="#666"
              value={responses[currentQuestion]?.toString() ?? ''}
              onChangeText={(text) => {
                const num = parseInt(text);
                if (!isNaN(num)) handleResponse(num);
              }}
            />
          </View>
        );

      case 'MCM':
        const selectedValues = (responses[currentQuestion] as string[]) ?? [];
        return (
          <View style={styles.optionsContainer}>
            {question.options?.map(option => (
              <Pressable
                key={option.id}
                style={({pressed}) => [
                  styles.optionButton,
                  selectedValues.includes(option.id.toString()) && styles.selectedOption,
                  pressed && styles.buttonPressed
                ]}
                onPress={() => {
                  const current = selectedValues;
                  const valueStr = option.id.toString();
                  const newValue = current.includes(valueStr)
                    ? current.filter(v => v !== valueStr)
                    : [...current, valueStr];
                  handleResponse(newValue);
                }}
              >
                <Text style={styles.optionText}>{option.text}</Text>
              </Pressable>
            ))}
          </View>
        );

      default: // MC and YN
        return (
          <View style={styles.optionsContainer}>
            {question.options?.map(option => (
              <Pressable
                key={option.id}
                style={({pressed}) => [
                  styles.optionButton,
                  responses[currentQuestion] === option.id && styles.selectedOption,
                  pressed && styles.buttonPressed
                ]}
                onPress={() => handleResponse(option.id)}
              >
                <Text style={styles.optionText}>{option.text}</Text>
              </Pressable>
            ))}
          </View>
        );
    }
  };

  const canProceed = () => {
    const response = responses[currentQuestion];
    if (response === undefined) return false;
    if (typeof response === 'string') return response.trim().length > 0;
    if (Array.isArray(response)) return response.length > 0;
    return true;
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>{error}</Text>
        <Pressable 
          style={styles.retryButton}
          onPress={fetchQuestions}
        >
          <Text style={styles.buttonText}>Retry</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Health Assessment</Text>
        <View style={styles.progressBar}>
          <View 
            style={[
              styles.progressFill,
              { width: `${((currentQuestion + 1) / questions.length) * 100}%` }
            ]} 
          />
        </View>
      </View>

      <ScrollView style={styles.content}>
        <Text style={styles.questionCount}>
          Question {currentQuestion + 1} of {questions.length}
        </Text>
        
        <Text style={styles.questionText}>
          {questions[currentQuestion]?.question}
        </Text>

        {renderQuestion()}
      </ScrollView>

      <View style={styles.footer}>
        <Pressable
          style={({pressed}) => [
            styles.nextButton,
            !canProceed() && styles.buttonDisabled,
            pressed && styles.buttonPressed
          ]}
          onPress={handleNext}
          disabled={!canProceed()}
        >
          <Text style={styles.buttonText}>
            {currentQuestion === questions.length - 1 ? 'Finish' : 'Next'}
          </Text>
        </Pressable>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
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
  progressBar: {
    height: 4,
    backgroundColor: '#333',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
  },
  content: {
    flex: 1,
    padding: 20,
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
  inputContainer: {
    marginBottom: 20,
  },
  textInput: {
    backgroundColor: '#1E1E1E',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 12,
    padding: 16,
    color: '#fff',
    fontSize: 16,
    height: 150,
    textAlignVertical: 'top',
  },
  numberInput: {
    backgroundColor: '#1E1E1E',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 12,
    padding: 16,
    color: '#fff',
    fontSize: 16,
  },
  optionsContainer: {
    gap: 12,
  },
  optionButton: {
    backgroundColor: '#1E1E1E',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  selectedOption: {
    backgroundColor: '#0A84FF',
    borderColor: '#0A84FF',
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
  footer: {
    padding: 16,
    backgroundColor: '#1E1E1E',
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  nextButton: {
    backgroundColor: '#0A84FF',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  retryButton: {
    backgroundColor: '#0A84FF',
    padding: 16,
    borderRadius: 12,
    marginHorizontal: 20,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  errorText: {
    color: '#ff4444',
    fontSize: 16,
    marginBottom: 16,
    textAlign: 'center',
  },
});

export default QC2;