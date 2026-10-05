questions = {
    'department': {
        'type': 'choice',
        'instructions': 'What should happen here?',
        'criteria': {
            'light on': 'When the user asks for a light to be turned on',
            'light off': 'When the user asks for a light to be turned off',
            'sports score': 'when user asks for information of ONE game',
            'light_brightness': 'When the user asks for the light to be turned to a certain brightness',
            'play audio': 'When the user asks for audio or music to be played',
            'pause audio': 'When the user asks for a pause in audio or music',
            'stop audio': 'When the user asks to stop playing audio or music',
            'resume audio': 'When the user asks to resumer playing audio or music',
            'temperature': 'use when user asks about temperature',
            'weather': 'use when user asks about weather',
            'humidity': 'use when user asks about humidity',
            'handoff to llm': 'multistep instructions, things not listed otherwise',
        },
    },
}