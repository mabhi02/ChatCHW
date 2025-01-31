import { View, StyleSheet } from 'react-native';

type ProgressProps = {
  value: number;
};

export const Progress = ({ value = 0 }: ProgressProps) => (
  <View style={styles.container}>
    <View style={[styles.bar, { width: `${value}%` }]} />
  </View>
);

const styles = StyleSheet.create({
  container: {
    height: 4,
    backgroundColor: '#e0e0e0',
    borderRadius: 2,
    overflow: 'hidden',
  },
  bar: {
    height: '100%',
    backgroundColor: '#007AFF',
  },
});