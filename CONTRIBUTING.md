Introduction
------------------------------

First off, thank you for your interest in contributing to this project. We aim to publish the software in an open-source software journal. For this a testing of our software and discussion of limitations by people outside of our research group at TU Berlin would be very helpful. In particular, we welcome newcomers and encourage diverse new contributors from all backgrounds.


Code of Conduct
------------------------------

Please be respectful, constructive, and professional in all interactions. We aim to maintain a welcoming, collaborative environment for everyone.


How to Contribute
------------------------------

> At the moment we would appropriate to receive contributions in the form of GitHub issues and in the GitHub discussion section, rather than pull requests by others than our research group.

We are in particular looking for feedback on the usability of our software, ideas for further improvements and extension of our LLM-based extraction workflow. 

> :flashlight: Please see the steps below as some guidelines to create a good issue or discussion:

> 1. Check if your issue or discussion has already been raised by searching the issues and discussions.
> 2. If your issue or discussion is new, please create a new issue or discussion and provide a clear and descriptive title.
> 3. Provide a detailed description of the issue or discussion, including any relevant context, and if needed, steps to reproduce the issue and a brief explanation of the expected vs. actual behavior. It is sufficient when you report the issue or suggestion from a user perspective. 
> 4. If applicable, you can also reference the respective code line(s) in your commit or discussion. Simply, go to the respective file (or the respective commit when it is not the most recent commit message), click on the code number, and select either a permalink (<- good when you have already opened an issue or discussion) or reference to a new issue or discussion. 
> 5. Maybe the most important point: keep in mind, every kind of feedback you provide or question you ask is valuable for improving the software project. Therefore, see the above points as ideas to help you to post an issue or discussion, but feel free to deviate from them at any time.


### When to use what
* **GitHub Discussions** are a good choice for questions about the project's purpose, intended use, workflow, outputs, limitations, or possible future directions. Discussions are also a good place to exchange ideas before deciding whether a concrete issue should be opened.

* **GitHub Issues** are a good choice if your question identifies a reproducible bug or a specific actionable problem. Either create a new Issue or link to an existing one


### Security
> :warning: If you find a security related issues, vulnerabilities or bugs including sensitive information, or elsewhere, please do not report them in public. Email anna.buch@tu-berlin.de instead.


How not to Contribute
------------------------------

External contributors should **not**:
- Submit pull requests.
- Submit patches or code changes through GitHub Issues or Discussions.
- Create forks for the purpose of submitting code back to this project.

Please describe the problem, proposed behavior, or idea instead. The project maintainers will decide whether and how to address it. Code contributions and pull requests are managed internally by the research group; external pull requests may be closed without review.


What happens after you submit feedback?
------------------------------

The developers may ask follow-up questions, label or categorize an issue. They may also close issues or comment on them before closing when they are outside the project's scope, or cannot be acted on. Please, keep in mind that we will consider feedback, but we cannot guarantee a response within a particular timeframe or that a proposed change will be implemented.

Thank you for helping us make the software clearer, more useful, and more reliable for research on infrastructure damages and cascaded societal, environmental, and economical impacts.



## Development Setup with uv

This project uses Python and `uv` for dependency management.

1. Clone the repository.
2. Change into the project directory.
3. Create the environment and install dependencies: `uv sync`
4. Activate the environment: `source .venv/bin/activate` (Mac, Linux) or `source .venv/Scripts/activate`(Windows)


>:flashlight: At this point, you're ready to make your changes! And as always, feel free to ask for help; either by opening an issue, wirtting something in the GitHub discussions, or by mailing: anna.buch@tu-berlin.de


Development setup with docker
----------------------------------

<< ------- under development ------- >>



Running the unit test suite
---------------------------

<< ------- under development ------- >>



Closing remark
-------------------
Some of the ideas and text snippets were taken from awesome open-source projects:
* [Celery](https://github.com/celery/celery/blob/main/CONTRIBUTING.rst#security)

