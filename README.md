A.D.A.M (A Digital Assistant Manager) 

A.D.A.M, standing for A Digital Assistant Manager, is an advanced digital assistant project designed for sophisticated user interactions through speech. This open-source project, developed by Chance Brownfield, offers a unique blend of features ranging from Automatic Multi-Speaker Speech Recognition (AMSSR) to sentiment analysis and modular scripting capabilities.

 Features

 Speech Recognition: Utilizes my custom AMSSR to capture and interpret user speech, enabling a dynamic conversational experience. Conversation Flow: The main script (main.py) orchestrates conversation flow, handling wake word detection, user and bot creation, and idle chat management. Modular Architecture: A.D.A.M is built with modularity in mind, featuring various scripts (Brain.py, response.py, Motor.py, etc.) to handle distinct functionalities.

 User Interaction: Brain.py processes user input, generates responses, and updates user and bot histories, adapting to emotional states through sentiment analysis (Limbic.py). Speech and Audio Processing: AMSSR.py seamlessly integrates speaker diarization, audio segmentation, and transcription for effective understanding and response to audio inputs.

 Scripting Capabilities: mod_builder.py facilitates the creation and management of Python mods, extending the assistant's capabilities through custom scripts. Please check te projoct_log.txt for more info on ADAM and its scripts.

Video Response Generation: The ability to generate video responses from prompts, responses, and images, using tools like Stable Video Diffusion, Pix2Pix, and pygame.

Information Extraction and Question Formulation: Pineal.py offers functionalities for web scraping, natural language processing, and system information gathering to provide a wide range of information extraction capabilities.

Emotion and Sentiment Analysis: Limbic.py provides a framework for analyzing emotions and sentiments in both text and audio inputs, generating responses based on user and bot profiles.

Profile Management: Functions from Hippocampus.py handle the creation and management of user and bot profiles, including user verification through voice recognition.

Idle Chat Handling: A mechanism to handle idle chat by selecting random wake words, retrieving unanswered questions from Reddit, and generating responses.

These features collectively make A.D.A.M a versatile and comprehensive system for voice interaction, information processing, and content generation.

 ADAM while entering its final stages is still a work in progress for example the AutoMod.py for SelfModding and Dynamic Command Generation is still being worked out and is currently unimplemented

 Getting Started Installation: Clone the repository and install the required dependencies using pip install -r requirements.txt. 

Configuration: Customize main.py and other configuration files according to your preferences and use case. 

Run A.D.A.M: Execute main.py to start A.D.A.M and engage in interactive conversations.

 Contribution Guidelines: We welcome contributions to enhance A.D.A.M's capabilities. Please contact me via email if you would like to contribute. 

License: A.D.A.M is currently under the Default License. 

Support: For any issues, feature requests, or general inquiries, please open an issue. Keep in mind that this is my first project I have no experience and no degrees this has been an up hill learning process for me even just setting up this repo was completely new to me so please be considerate of my inexperience. With that said I do not mind constuctive critisism after all I am learning so feel free to point out anything I could have done better or even better feel free to contact me directly to contribute yourself ADAM is designed to be extended Let's build a smarter digital assistant together! 🤖🗣️ 

Honorary Contributors: If you would like to help ADAM's development but can't contibute code you can still help by supporting the project and becoming an Honorary Contributor please sponsor to get a spot on this list ()
