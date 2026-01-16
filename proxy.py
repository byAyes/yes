from flask import Flask, request, jsonify, Response
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

# Configuration
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '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')
NVIDIA_BASE_URL = os.environ.get('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')

# Model mapping (OpenAI model names to NVIDIA NIM models)
MODEL_MAPPING = {
    'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
    'gpt-4': 'meta/llama-3.1-70b-instruct',
    'gpt-4-turbo': 'meta/llama-3.1-405b-instruct',
}

def map_model(openai_model):
    """Map OpenAI model names to NVIDIA NIM models"""
    return MODEL_MAPPING.get(openai_model, openai_model)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        
        # Map the model name
        original_model = data.get('model', 'gpt-3.5-turbo')
        nvidia_model = map_model(original_model)
        
        # Prepare NVIDIA NIM request
        nvidia_payload = {
            'model': nvidia_model,
            'messages': data.get('messages', []),
            'temperature': data.get('temperature', 0.7),
            'top_p': data.get('top_p', 1.0),
            'max_tokens': data.get('max_tokens', 1024),
            'stream': data.get('stream', False)
        }
        
        # Add optional parameters if present
        if 'frequency_penalty' in data:
            nvidia_payload['frequency_penalty'] = data['frequency_penalty']
        if 'presence_penalty' in data:
            nvidia_payload['presence_penalty'] = data['presence_penalty']
        
        headers = {
            'Authorization': f'Bearer {NVIDIA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Handle streaming
        if nvidia_payload['stream']:
            return handle_streaming(nvidia_payload, headers, original_model)
        else:
            return handle_non_streaming(nvidia_payload, headers, original_model)
            
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'proxy_error',
                'code': 500
            }
        }), 500

def handle_non_streaming(nvidia_payload, headers, original_model):
    """Handle non-streaming requests"""
    response = requests.post(
        f'{NVIDIA_BASE_URL}/chat/completions',
        headers=headers,
        json=nvidia_payload
    )
    
    if response.status_code != 200:
        return jsonify({
            'error': {
                'message': response.text,
                'type': 'nvidia_api_error',
                'code': response.status_code
            }
        }), response.status_code
    
    nvidia_response = response.json()
    
    # Convert NVIDIA response to OpenAI format
    openai_response = {
        'id': nvidia_response.get('id', 'chatcmpl-' + str(int(datetime.now().timestamp()))),
        'object': 'chat.completion',
        'created': int(datetime.now().timestamp()),
        'model': original_model,
        'choices': nvidia_response.get('choices', []),
        'usage': nvidia_response.get('usage', {})
    }
    
    return jsonify(openai_response)

def handle_streaming(nvidia_payload, headers, original_model):
    """Handle streaming requests"""
    def generate():
        response = requests.post(
            f'{NVIDIA_BASE_URL}/chat/completions',
            headers=headers,
            json=nvidia_payload,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    if data_str.strip() == '[DONE]':
                        yield f'data: [DONE]\n\n'
                        break
                    
                    try:
                        data = json.loads(data_str)
                        # Convert to OpenAI format if needed
                        data['model'] = original_model
                        yield f'data: {json.dumps(data)}\n\n'
                    except json.JSONDecodeError:
                        continue
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models in OpenAI format"""
    models = []
    for openai_model, nvidia_model in MODEL_MAPPING.items():
        models.append({
            'id': openai_model,
            'object': 'model',
            'created': int(datetime.now().timestamp()),
            'owned_by': 'nvidia'
        })
    
    return jsonify({
        'object': 'list',
        'data': models
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print('Starting NVIDIA NIM to OpenAI API Proxy...')
    print(f'Using NVIDIA Base URL: {NVIDIA_BASE_URL}')
    print('Available model mappings:')
    for openai_model, nvidia_model in MODEL_MAPPING.items():
        print(f'  {openai_model} -> {nvidia_model}')
    
    # Run on all interfaces so it's accessible from Android
    app.run(host='0.0.0.0', port=5000, debug=False)