Exercises:
* [ggml-test](ggml-test) - A simple test program using ggml to load model weights trained using PyTorch, and run using ggml.
* [cuda-test](cuda) - A simple cuda program to add two float16 vectors.
* [ddpm.py](ddpm.py) - Simple non-latent DDPM implementation. Works, and generates butterfly images from the Smithsonian dataset.
* [latent_ddpm.py](latent_ddpm.py) - Simple latent DDPM implementation. Runs, but generates garbage outputs. Something's not right.
* [toy_attention.py](toy_attention.py) - Simple multi-headed self-attention implementation. Runs, but isn't sufficient to give good results. Training saturates at a loss value that's still pretty much garbage.
* [toy_attention2.py](toy_attention2.py) - Training the multi-headed self-attention implementation to classify a few simple strings as Yes/No. This works well, and I intentionally over-fitted the model in order to analyse the behavior of the key/query/value vectors and attention outputs.
* [video_dream_vae.py](video_dream_vae.py) - Video per-frame encode and decode using a vae, with an imagination parameter to control how much it tries to replicate the original video. Poor results, just adds random artifacts to the frames instead of different ideas.
* [video_dream_sd.py](video_dream_sd.py) - Video per-frame encode and decode using Stable Diffusion, with an imagination parameter to control how much it tries to replicate the original video. Unstable janky results.
* [botchat.py](botchat.py) - Make two bots chat with each other. You can describe the two bots, and the scene, and it'll call an OpenAI-compatible API to make the two bots converse with each other. It's interesting, but the conversation "stagnates" after 20-30 messages and becomes repetitive. It works with locally-hosted LLMs as well (e.g. using LMStudio).
* [summarize-bookmarklet.js](summarize-bookmarklet.js) - Bookmarklet for a browser that summarizes the selected text on a webpage. It'll call an OpenAI-compatible API, and works with locally-hosted LLMs as well (e.g. using LMStudio).
