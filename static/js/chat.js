/**
 * chat interface
 */

class ChatInterface {
    constructor() {
        this.chatContainer = document.getElementById('chatContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.modelInfo = document.getElementById('modelInfo');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.temperatureSlider = document.getElementById('temperature');
        this.temperatureValue = document.getElementById('temperatureValue');
        this.maxLengthInput = document.getElementById('maxLength');
        
        this.isConnected = false;
        this.isTyping = false;
        
        this.init();
    }
    
    init() {
        // event listeners
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => this.handleKeyPress(e));
        this.temperatureSlider.addEventListener('input', (e) => this.updateTemperatureDisplay(e));
        
        // initialize controls
        this.updateTemperatureDisplay();
        
        // inital status check
        this.checkServerStatus();
        
        // periodic status check
        setInterval(() => this.checkServerStatus(), 30000); // Check every 30 seconds
        
        // Focus input
        this.messageInput.focus();
        
        // welcome message
        this.addMessage('system', 'Welcome to tank-gpt, what do you want to talk about?');
    }
    
    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
    
    updateTemperatureDisplay() {
        const value = this.temperatureSlider.value;
        this.temperatureValue.textContent = value;
    }
    
    async checkServerStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            this.isConnected = response.ok;
            this.updateStatus(data.model_loaded, data.model_type);
            
        } catch (error) {
            console.error('Health check failed:', error);
            this.isConnected = false;
            this.updateStatus(false, 'unknown');
        }
    }
    
    updateStatus(modelLoaded, modelType) {
        if (modelLoaded && this.isConnected) {
            this.statusIndicator.className = 'status-indicator online';
            this.modelInfo.textContent = `Model: ${modelType.toUpperCase()} (Ready)`;
            this.sendButton.disabled = false;
        } else if (this.isConnected) {
            this.statusIndicator.className = 'status-indicator offline';
            this.modelInfo.textContent = 'Model: Not Loaded';
            this.sendButton.disabled = true;
        } else {
            this.statusIndicator.className = 'status-indicator offline';
            this.modelInfo.textContent = 'Server: Disconnected';
            this.sendButton.disabled = true;
        }
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        this.addMessage('user', message);
        this.messageInput.value = '';
        
        const typingId = this.addMessage('bot', 'Typing...', true);
        this.isTyping = true;
        this.sendButton.disabled = true;
        
        try {
            const requestBody = {
                message: message,
                temperature: parseFloat(this.temperatureSlider.value),
                max_length: parseInt(this.maxLengthInput.value)
            };
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            
            this.removeMessage(typingId);
            
            if (response.ok) {
                const data = await response.json();
                this.addMessage('bot', data.response);
                this.updateStatus(true, data.model_type);
            } else {
                const error = await response.json();
                this.addMessage('error', `Error: ${error.detail || 'Unknown error occurred'}`);
            }
            
        } catch (error) {
            console.error('Chat request failed:', error);
            this.removeMessage(typingId);
            this.addMessage('error', 'Error: Failed to connect to server');
            this.isConnected = false;
            this.updateStatus(false, 'unknown');
        } finally {
            this.isTyping = false;
            this.sendButton.disabled = !this.isConnected;
            this.messageInput.focus();
        }
    }
    
    addMessage(type, content, isTyping = false) {
        const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.id = messageId;
        
        let userName = '';
        if (type === 'user') {
            userName = 'You';
        } else if (type === 'bot') {
            userName = 'AI Assistant';
        } else if (type === 'error') {
            userName = 'System';
        } else if (type === 'system') {
            userName = 'Tank GPT';
        }
        
        const contentHtml = isTyping 
            ? `<strong>${userName}:</strong> <span class="loading-indicator">${content}</span>`
            : `<strong>${userName}:</strong> ${this.escapeHtml(content)}`;
        
        messageDiv.innerHTML = contentHtml;
        
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageId;
    }
    
    removeMessage(messageId) {
        const messageElement = document.getElementById(messageId);
        if (messageElement) {
            messageElement.remove();
        }
    }
    
    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // utility methods for external control
    clearChat() {
        this.chatContainer.innerHTML = '';
        this.addMessage('system', 'Chat cleared. Start a new conversation!');
    }
    
    setTemperature(value) {
        this.temperatureSlider.value = value;
        this.updateTemperatureDisplay();
    }
    
    setMaxLength(value) {
        this.maxLengthInput.value = value;
    }
}

// initialize the chat interface when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});

// export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatInterface;
}
