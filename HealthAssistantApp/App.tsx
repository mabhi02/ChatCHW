import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import QC2 from './components/QC2';
import Chat2 from './components/Chat2';

// Define the type for screening answers
export type ScreeningAnswer = {
  question: string;
  answer: string | number | string[];
};

// Define the type for the navigation stack parameters
export type RootStackParamList = {
  QC2: undefined;
  Chat2: {
    screeningAnswers: ScreeningAnswer[];
    initialResponse: string;
    conversationId: string;
  };
};

// Create the stack navigator with the correct type
const Stack = createStackNavigator<RootStackParamList>();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="QC2"
        screenOptions={{
          headerShown: false,
          cardStyle: { backgroundColor: '#121212' }
        }}
      >
        <Stack.Screen 
          name="QC2" 
          component={QC2}
        />
        <Stack.Screen 
          name="Chat2" 
          component={Chat2}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;