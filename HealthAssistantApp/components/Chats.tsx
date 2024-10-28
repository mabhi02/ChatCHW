import React, { useState, useEffect, useRef } from 'react';
import { 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  FlatList, 
  StyleSheet, 
  KeyboardAvoidingView, 
  Platform,
  SafeAreaView,
  ActivityIndicator,
  Modal,
  Alert,
} from 'react-native';
import { RouteProp, useNavigation } from '@react-navigation/native';
import { RootStackParamList } from '../App';
import Icon from 'react-native-vector-icons/Ionicons';

// Define the type for the route parameters
type ChatsProps = {
  route: RouteProp<RootStackParamList, 'Chats'>;
};

// API endpoint for localhost
const API_URL = 'http://localhost:5000';

// Message type definition
type Message = {
  id: string;
  text: string;
  sender: 'user' | 'bot';
};

// Chats Component
const Chats: React.FC<ChatsProps> = ({ route }) => {
  const { screeningAnswers, initialResponse, conversationId } = route.params;
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<'followup' | 'test' | 'advice' | 'complete'>('followup');
  const [modalVisible, setModalVisible] = useState(false);
  const [fullConversation, setFullConversation] = useState<Message[]>([]);
  const flatListRef = useRef<FlatList>(null);
  const navigation = useNavigation();

  useEffect(() => {
    if (initialResponse) {
      const initialMsg: Message = { id: generateUniqueId(), text: initialResponse, sender: 'bot' };
      setMessages([initialMsg]);
    }
    displayScreeningAnswers();
  }, []);

  // Display screening answers as a bot message
  const displayScreeningAnswers = () => {
    if (screeningAnswers && screeningAnswers.length > 0) {
      const summaryText = screeningAnswers.map(answer => 
        `${answer.question}: ${answer.answer}`
      ).join('\n');
      const summaryMessage: Message = { id: generateUniqueId(), text: `Screening Summary:\n${summaryText}`, sender: 'bot' };
      setMessages(prev => [...prev, summaryMessage]);
    }
  };

  // Automatically scroll to the bottom when messages change
  useEffect(() => {
    if (flatListRef.current) {
      flatListRef.current.scrollToEnd({ animated: true });
    }
  }, [messages]);

  // Handle sending a message
  const handleSend = async () => {
    if (input.trim() === '') return;

    const newMessage: Message = { id: generateUniqueId(), text: input, sender: 'user' };
    setMessages(prev => [...prev, newMessage]);
    setInput('');

    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          screeningAnswers,
          conversationHistory: [...messages, newMessage],
          conversationId,
          stage,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botMessage: Message = { id: generateUniqueId(), text: data.response, sender: 'bot' };
      setMessages(prev => [...prev, botMessage]);
      updateStage();
    } catch (error) {
      console.error('Error sending message:', error);
      Alert.alert(
        'Error',
        'Failed to send message. Please check your internet connection and try again.',
        [{ text: 'OK' }]
      );
      // Optionally, add an error message to the chat
      const errorMessage: Message = { 
        id: generateUniqueId(), 
        text: 'Sorry, something went wrong. Please try again.', 
        sender: 'bot' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // Update the conversation stage based on the number of messages
  const updateStage = () => {
    setStage(prevStage => {
      switch (prevStage) {
        case 'followup':
          return messages.length >= 7 ? 'test' : 'followup';
        case 'test':
          return messages.length >= 9 ? 'advice' : 'test';
        case 'advice':
          return 'complete';
        default:
          return prevStage;
      }
    });
  };

  // Handle viewing the full conversation in a modal
  const handleViewFullConversation = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/full_conversation/${conversationId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setFullConversation(data.messages);
      setModalVisible(true);
    } catch (error) {
      console.error('Error fetching full conversation:', error);
      Alert.alert(
        'Error',
        'Failed to fetch the full conversation. Please try again later.',
        [{ text: 'OK' }]
      );
    } finally {
      setLoading(false);
    }
  };

  // Render each message bubble
  const renderMessage = ({ item }: { item: Message }) => (
    <View style={[
      styles.messageBubble, 
      item.sender === 'user' ? styles.userBubble : styles.botBubble
    ]}>
      <Text style={styles.messageText}>{item.text}</Text>
    </View>
  );

  // Simple unique ID generator
  const generateUniqueId = () => {
    return Math.random().toString(36).substr(2, 9);
  };

  return (
    <SafeAreaView style={styles.container}>
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.flatListContent}
        ListHeaderComponent={
          <TouchableOpacity onPress={handleViewFullConversation} style={styles.viewFullButton}>
            <Text style={styles.viewFullButtonText}>View Full Conversation</Text>
          </TouchableOpacity>
        }
        showsVerticalScrollIndicator={false}
        style={styles.flatList}
      />

      <KeyboardAvoidingView 
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.inputWrapper}
        keyboardVerticalOffset={Platform.OS === "ios" ? 60 : 0}
      >
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholder="Type a message..."
            placeholderTextColor="#888"
            multiline
            editable={!loading && stage !== 'complete'}
          />
          <TouchableOpacity 
            onPress={handleSend}
            disabled={loading || stage === 'complete'}
            style={styles.sendButton}
          >
            {loading ? (
              <ActivityIndicator color="#1E88E5" />
            ) : (
              <Icon name="send" size={24} color={ (loading || stage === 'complete') ? '#888' : '#1E88E5'} />
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>

      <Modal
        animationType="slide"
        transparent={false}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <SafeAreaView style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Full Conversation</Text>
            <TouchableOpacity onPress={() => setModalVisible(false)} style={styles.closeButton}>
              <Icon name="close" size={24} color="#FFFFFF" />
            </TouchableOpacity>
          </View>
          <FlatList
            data={fullConversation}
            renderItem={renderMessage}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.modalFlatListContent}
            showsVerticalScrollIndicator={false}
            style={styles.modalFlatList}
          />
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
};

// Styles remain the same
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
  },
  flatList: {
    flex: 1,
    paddingHorizontal: 10,
    paddingTop: 10,
  },
  flatListContent: {
    paddingBottom: 20,
  },
  viewFullButton: {
    backgroundColor: '#3700B3',
    padding: 10,
    marginBottom: 10,
    borderRadius: 5,
    alignItems: 'center',
  },
  viewFullButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  messageBubble: {
    padding: 10,
    borderRadius: 10,
    marginBottom: 10,
    maxWidth: '80%',
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#1E88E5',
  },
  botBubble: {
    alignSelf: 'flex-start',
    backgroundColor: '#424242',
  },
  messageText: {
    color: '#FFFFFF',
  },
  inputWrapper: {
    backgroundColor: '#1E1E1E',
    padding: 10,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#444',
    padding: 10,
    marginRight: 10,
    borderRadius: 20,
    color: '#FFFFFF',
    backgroundColor: '#222',
    maxHeight: 100,
  },
  sendButton: {
    padding: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#121212',
  },
  modalHeader: {
    height: 60,
    backgroundColor: '#1E88E5',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 10,
  },
  modalTitle: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: 'bold',
  },
  closeButton: {
    padding: 5,
  },
  modalFlatList: {
    flex: 1,
    padding: 10,
  },
  modalFlatListContent: {
    paddingBottom: 20,
  },
});

export default Chats;