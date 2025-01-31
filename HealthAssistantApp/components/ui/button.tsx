import { TouchableOpacity, Text, StyleSheet, TouchableOpacityProps } from 'react-native';

type ButtonProps = TouchableOpacityProps & {
  onPress: () => void;
  children: React.ReactNode;
  variant?: 'default' | 'outline';
  disabled?: boolean;
};

export const Button = ({ 
  onPress, 
  children, 
  variant = 'default', 
  disabled = false, 
  style = {} 
}: ButtonProps) => (
  <TouchableOpacity 
    onPress={onPress}
    disabled={disabled}
    style={[
      styles.button,
      variant === 'outline' && styles.outlineButton,
      disabled && styles.disabled,
      style
    ]}
  >
    <Text style={[
      styles.text,
      variant === 'outline' && styles.outlineText,
      disabled && styles.disabledText
    ]}>
      {children}
    </Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  outlineButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  disabled: {
    backgroundColor: '#ccc',
  },
  text: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  outlineText: {
    color: '#007AFF',
  },
  disabledText: {
    color: '#666',
  },
});