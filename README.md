# 🚀 Smarter Reconciliation and Anomaly Detection

## 📌 Table of Contents
- [Introduction](#introduction)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
This project aims to streamline the reconciliation process by implementing an anomaly detection system using Generative AI. It addresses the challenge of manually identifying data anomalies in large transaction datasets, which is often tedious and error-prone. By leveraging historical data and LLMs, we automate anomaly detection and provide insights into potential root causes, ultimately reducing manual effort and minimizing human error.

## ⚙️ What It Does
-   Automatically detects data anomalies by comparing real-time data against historical baselines.
-   Provides categorized insights into potential root causes of detected anomalies (predefined buckets).
-   Integrates with existing reconciliation tools to streamline the anomaly identification process.
-   Allows reconcilers to provide feedback on detected anomalies for future refinement.
-   Utilizes Agentic AI to provide concise summaries of break resolutions.
-   Streamlines workflow with Operator Assist Agents for task creation, API calls, email sending, and ticket creation.
-   Outputs anomaly detection results based on reconciliation details, current, and historical data.

## 🛠️ How We Built It
We utilized LLMs such as OpenAI's GPT, along with techniques like clustering and anomaly detection, to identify patterns and validate data consistency. We developed a classification system for anomaly reasons, and implemented an interactive tool for reconciler feedback. Agentic AI was employed to summarize break resolutions and automate operator tasks.

## 🚧 Challenges We Faced
-   Developing a robust classification system for anomaly reasons within predefined buckets.
-   Implementing effective techniques for identifying patterns and validating data consistency.
-   Integrating Agentic AI to accurately summarize break resolutions and automate operator tasks.
-   Handling and processing large volumes of historical and real-time reconciliation data efficiently.
-   Creating an interactive feedback tool that effectively refines anomaly detection accuracy.

## 🏃 How to Run
1. Clone the repository 
   ```sh
   git clone [https://github.com/ewfx/sradg-incident-creator](https://github.com/ewfx/sradg-incident-creator)
   ```
2. Install dependencies 
   ```sh
   pip install -r requirements.txt
   ```
3. Run the project 
   ```sh
   python ui.py
   ```

## 🏗️ Tech Stack
-   🔹 Backend: Python, FastAPI
-   🔹 Database: PostgreSQL
-   🔹 Other: LLM - Llama, Ollama

## 👥 Team
-   Gyanendra Shukla - [GitHub](https://github.com/srirajshukla/)
-   Aniket Shandilya - [GitHub](https://github.com/Aniket62058/)
```
