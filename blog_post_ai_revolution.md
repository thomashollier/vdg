# The Real AI Revolution in VFX and Film Production

AI has hit film production and the VFX industry with lightning speed, and we haven't had the time to figure out how it can change the way we work. From a legacy VFX perspective, the instinct is to take a model-specific view. We select specific generative AI models and learn to coax them into producing a desired output, one shot at a time. That's understandable, but the monolithic nature of models and the speed at which they change calls for a lighter, more flexible way of working. It's not about finding perfect models that solve everything. It's about using the power of large language models to quickly assemble, test, and operate custom pipelines from the best AI or traditional tools available at any particular time, and change them just as quickly when something better comes along.

## Legacy Pipelines are Rigid by Design

Every VFX artist knows this reality. You have Maya for animation. Nuke for compositing. Substance for textures. Photoshop for matte painting. Unreal Engine for real-time rendering. Each tool is powerful on its own, but they're difficult to connect together in a consistent and flexible manner. You also need teams of specialists with the unique skillsets to handle each step. Large studios solved this by building massive pipelines around fixed toolsets, and succeeded by enforcing a way of working. This is the opposite of what generative AI requires because the models are evolving too fast and the old approach can't adapt quickly enough.

## VDG: A Case Study

Over the past few years, I developed various processes to transform video clips into still art prints. Even though I'm not a programmer, I coded a bunch of Python scripts from scratch, generating and combining data from multiple freely available tools and libraries. My code was messy. Every time I thought of another cool thing to try, I had to spend a few days shoehorning it into the work I'd already done. I knew a specific tool could do what I needed and I understood what it needed to do its work, but integrating new functionality without breaking existing workflows kept me from freely exploring every crazy idea that popped into my head.

I eventually decided to see how far I could get with an LLM. The result is VDG—a node-based system for video processing that pulls together OpenCV, Blender, NumPy, FFmpeg, FastAPI, and exiftool. None of these tools know about each other natively, but in VDG they're connected through a visual node graph where I can wire up a complete pipeline without writing code.

## You Still Need to Know the Process

The one thing a coding assistant can't do for you is understand what you want if you can't describe it. You're the architect and the assistant is a very fast builder who needs good blueprints. I came to VDG with years of doing this work manually and my bad Python code illustrated what each step was designed to produce and how the data flowed from one stage to the next. Then, based on my knowledge of UI's and workflows, I described with words how I wanted the functionality in my linear ad-hoc scripts to be reimplemented in a clean and modular manner.

## The Flexibility Advantage

Last week I realized OpenCV's tracking wasn't giving me the control I needed. In the past, I'd had more success with Blender's tracking tools but required a round trip manually importing, exporting, and converting this data right in the middle of my automated workflow. The assistant helped me write a bridge script that launches Blender, lets me track manually, then exports the data back into VDG's format. I described what I had in mind and the assistant had the necessary understanding to implement the request into the existing architecture.

This is the flexibility that matters. Not "AI generates my shots" but "AI helps me build and modify tools fast enough to keep up with my creative needs."

## The Point

This opens up a different way of working. The footprint is lighter. The turnaround is faster. And when a better component comes along, you can swap it in without rebuilding from scratch. That kind of flexibility matters whether you're working on a single project or managing workflows across dozens.

Generative AI models will keep producing more impressive results but the solution isn't using this or that latest model. Productions that need reliable, controllable, production-ready workflows, the bigger story is the possibility of a flexible infrastructure layer: solving your specific problems, your way, using whatever combination of tools works best right now—with the freedom to pivot quickly.
