import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import Questionnaire from './components/Questionnaire';
import Chats from './components/Chats';

export type ScreeningAnswer = { question: string; answer: string | number | string[] };

export type RootStackParamList = {
  Questionnaire: undefined;
  Chats: { 
    screeningAnswers: ScreeningAnswer[]; 
    initialResponse?: string;
    conversationId?: string;
  };
};

const Stack = createStackNavigator<RootStackParamList>();

const App: React.FC = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator 
        initialRouteName="Questionnaire"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#1E88E5',
          },
          headerTintColor: '#FFFFFF',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen 
          name="Questionnaire" 
          component={Questionnaire} 
          options={{ title: 'Health Questionnaire' }}
        />
        <Stack.Screen 
          name="Chats" 
          component={Chats} 
          options={{ title: 'Health Assistant Chat' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;