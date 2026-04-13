from flask import Flask, request, jsonify, Response
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Configuration
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', 'your-nvidia-api-key-here')
NVIDIA_BASE_URL = os.environ.get('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
PORT = int(os.environ.get('PORT', 5000))

# Model mapping (OpenAI model names to NVIDIA NIM models)
MODEL_MAPPING = {
    'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
    'gpt-4': 'meta/llama-3.1-70b-instruct',
    'gpt-4-turbo': 'meta/llama-3.1-405b-instruct',
    'deepseek-chat': 'deepseek-ai/deepseek-r1',
    'deepseek-r1': 'deepseek-ai/deepseek-r1',
    'deepseek-coder': 'deepseek-ai/deepseek-coder-6.7b-instruct',
    'deepseek-v3.1': 'deepseek-ai/deepseek-v3.1',
    'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2',
    'gemma-4': 'google/gemma-4-31b-it',
    'glm-4.7': 'z-ai/glm4_7',
    'glm-5': 'z-ai/glm5',
}

# Image model mapping (OpenAI image models to NVIDIA NIM image models)
IMAGE_MODEL_MAPPING = {
    'dall-e-2': 'stabilityai/stable-diffusion-xl',
    'dall-e-3': 'black-forest-labs/flux-1-dev',
    'stable-diffusion-xl': 'stabilityai/stable-diffusion-xl',
    'flux-1-dev': 'black-forest-labs/flux-1-dev',
    'flux-1-schnell': 'black-forest-labs/flux-1-schnell',
}

def map_model(openai_model):
    """Map OpenAI model names to NVIDIA NIM models"""
    return MODEL_MAPPING.get(openai_model, openai_model)

def map_image_model(openai_model):
    """Map OpenAI image model names to NVIDIA NIM image models"""
    return IMAGE_MODEL_MAPPING.get(openai_model, openai_model)

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        print(f"Received request: {json.dumps(data, indent=2)}")
        
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
    print(f"Sending to NVIDIA: {json.dumps(nvidia_payload, indent=2)}")
    
    try:
        response = requests.post(
            f'{NVIDIA_BASE_URL}/chat/completions',
            headers=headers,
            json=nvidia_payload,
            timeout=60
        )
        
        print(f"NVIDIA response status: {response.status_code}")
        print(f"NVIDIA response: {response.text[:500]}")
        
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
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        return jsonify({
            'error': {
                'message': f'Failed to connect to NVIDIA API: {str(e)}',
                'type': 'connection_error',
                'code': 500
            }
        }), 500

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
    
    for openai_model, nvidia_model in IMAGE_MODEL_MAPPING.items():
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

@app.route('/v1/images/generations', methods=['POST', 'OPTIONS'])
def image_generations():
    """OpenAI-compatible image generation endpoint"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        print(f"Received image request: {json.dumps(data, indent=2)}")
        
        # Map the model name
        original_model = data.get('model', 'dall-e-3')
        nvidia_model = map_image_model(original_model)
        
        # Get parameters
        prompt = data.get('prompt', '')
        n = data.get('n', 1)  # number of images
        size = data.get('size', '1024x1024')
        
        # Parse size
        width, height = 1024, 1024
        if 'x' in size:
            width, height = map(int, size.split('x'))
        
        # Prepare NVIDIA NIM request
        nvidia_payload = {
            'text_prompts': [{'text': prompt, 'weight': 1}],
            'cfg_scale': 7.5,
            'sampler': 'K_EULER_ANCESTRAL',
            'samples': 1,
            'steps': 50
        }
        
        # Add size parameters
        if width:
            nvidia_payload['width'] = width
        if height:
            nvidia_payload['height'] = height
        
        headers = {
            'Authorization': f'Bearer {NVIDIA_API_KEY}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        print(f"Sending to NVIDIA image API: {json.dumps(nvidia_payload, indent=2)}")
        
        # Generate images
        images = []
        for i in range(n):
            try:
                response = requests.post(
                    f'{NVIDIA_BASE_URL}/{nvidia_model}',
                    headers=headers,
                    json=nvidia_payload,
                    timeout=120
                )
                
                print(f"NVIDIA image response status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"NVIDIA image error: {response.text[:500]}")
                    continue
                
                nvidia_response = response.json()
                
                # NVIDIA returns base64 encoded image
                if 'image' in nvidia_response:
                    images.append({
                        'b64_json': nvidia_response['image'],
                        'revised_prompt': prompt
                    })
                elif 'data' in nvidia_response and len(nvidia_response['data']) > 0:
                    images.append({
                        'b64_json': nvidia_response['data'][0].get('b64_json', ''),
                        'revised_prompt': prompt
                    })
                    
            except Exception as e:
                print(f"Error generating image {i+1}: {str(e)}")
                continue
        
        if not images:
            return jsonify({
                'error': {
                    'message': 'Failed to generate images',
                    'type': 'image_generation_error',
                    'code': 500
                }
            }), 500
        
        # Convert to OpenAI format
        openai_response = {
            'created': int(datetime.now().timestamp()),
            'data': images
        }
        
        return jsonify(openai_response)
        
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'proxy_error',
                'code': 500
            }
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API info"""
    return jsonify({
        'name': 'NVIDIA NIM to OpenAI API Proxy',
        'version': '1.0.0',
        'endpoints': {
            'chat': '/v1/chat/completions',
            'images': '/v1/images/generations',
            'models': '/v1/models',
            'health': '/health'
        },
        'status': 'running'
    })

if __name__ == '__main__':
    print('Starting NVIDIA NIM to OpenAI API Proxy...'); print(f'Using NVIDIA Base URL: {NVIDIA_BASE_URL}'); print(f'Running on port: {PORT}'); print('Available chat model mappings:'); [print(f'  {k} -> {v}') for k, v in MODEL_MAPPING.items()]; print('Available image model mappings:'); [print(f'  {k} -> {v}') for k, v in IMAGE_MODEL_MAPPING.items()]
    
    # Run on all interfaces so it's accessible from Android
    app.run(host='0.0.0.0', port=PORT, debug=False)