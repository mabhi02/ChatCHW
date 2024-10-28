import React, { useState, useEffect } from 'react';
import { View, Text, Button, TextInput, ScrollView, StyleSheet, Alert } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList, ScreeningAnswer } from '../App';

type QuestionnaireProps = {
  navigation: StackNavigationProp<RootStackParamList, 'Questionnaire'>;
};

type Question = {
  question: string;
  type: 'MC' | 'NUM' | 'YN' | 'MCM' | 'FREE';
  options?: { id: number | string; text: string }[];
  range?: { min: number; max: number; step: number; unit: string };
};

const API_URL = 'http://localhost:5000';

const Questionnaire: React.FC<QuestionnaireProps> = ({ navigation }) => {
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
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setQuestions(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching questions:', error);
      setError('Failed to load questions. Please try again.');
      setLoading(false);
    }
  };

  const handleInputChange = (value: string | number | string[]) => {
    const newResponses = [...responses];
    newResponses[currentQuestion] = value;
    setResponses(newResponses);
  };

  const handleNextQuestion = async () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      const screeningAnswers: ScreeningAnswer[] = questions.map((q, index) => ({
        question: q.question,
        answer: responses[index]
      }));

      try {
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

        if (!chatResponse.ok) {
          throw new Error(`HTTP error! status: ${chatResponse.status}`);
        }

        const chatData = await chatResponse.json();
        navigation.navigate('Chats', { 
          screeningAnswers, 
          initialResponse: chatData.response,
          conversationId: chatData.conversationId
        });
      } catch (error) {
        console.error('Error:', error);
        Alert.alert('Error', 'There was an error starting the chat. Please try again.');
      }
    }
  };

  const renderQuestion = () => {
    if (questions.length === 0) return null;
    const question = questions[currentQuestion];
    switch (question.type) {
      case 'MC':
      case 'YN':
        return (
          <Picker
            selectedValue={responses[currentQuestion]}
            onValueChange={(itemValue) => handleInputChange(itemValue)}
            style={styles.picker}
          >
            {question.options?.map((option) => (
              <Picker.Item key={option.id} label={option.text} value={option.id} color="#FFFFFF" />
            ))}
          </Picker>
        );
      case 'NUM':
        return (
          <TextInput
            style={styles.input}
            keyboardType="numeric"
            onChangeText={(text) => handleInputChange(parseInt(text) || 0)}
            value={responses[currentQuestion]?.toString()}
            placeholderTextColor="#888"
          />
        );
      case 'MCM':
        return (
          <View>
            {question.options?.map((option) => (
              <Button
                key={option.id}
                title={option.text}
                onPress={() => {
                  const currentValue = responses[currentQuestion] as string[] || [];
                  const newValue = currentValue.includes(option.id.toString())
                    ? currentValue.filter(id => id !== option.id.toString())
                    : [...currentValue, option.id.toString()];
                  handleInputChange(newValue);
                }}
                color={
                  (responses[currentQuestion] as string[] || []).includes(option.id.toString())
                    ? '#1E88E5'
                    : '#424242'
                }
              />
            ))}
          </View>
        );
      case 'FREE':
        return (
          <TextInput
            style={styles.input}
            multiline
            numberOfLines={4}
            onChangeText={(text) => handleInputChange(text)}
            value={responses[currentQuestion]?.toString()}
            placeholderTextColor="#888"
          />
        );
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Loading questions...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Error: {error}</Text>
        <Button title="Retry" onPress={fetchQuestions} color="#1E88E5" />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.question}>{questions[currentQuestion]?.question}</Text>
      {renderQuestion()}
      <Button
        title={currentQuestion < questions.length - 1 ? 'Next' : 'Finish'}
        onPress={handleNextQuestion}
        color="#1E88E5"
      />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#121212',
  },
  question: {
    fontSize: 18,
    marginBottom: 10,
    color: '#FFFFFF',
  },
  text: {
    color: '#FFFFFF',
  },
  input: {
    borderWidth: 1,
    borderColor: '#444',
    padding: 10,
    marginBottom: 10,
    borderRadius: 5,
    color: '#FFFFFF',
    backgroundColor: '#222',
  },
  picker: {
    backgroundColor: '#222',
    color: '#FFFFFF',
  },
});

export default Questionnaire;