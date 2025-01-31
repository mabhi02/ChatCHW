import { View, StyleSheet, ViewProps } from 'react-native';

type CardProps = ViewProps & {
  children: React.ReactNode;
};

export const Card = ({ children, style = {} }: CardProps) => (
  <View style={[styles.card, style]}>
    {children}
  </View>
);

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    margin: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
});