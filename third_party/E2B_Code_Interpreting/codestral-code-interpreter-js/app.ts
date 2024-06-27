import fs from 'node:fs'
import { CodeInterpreter, Result, ProcessMessage } from '@e2b/code-interpreter'
import * as dotenv from 'dotenv'
import MistralClient from '@mistralai/mistralai'

dotenv.config()

const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY || ''
const E2B_API_KEY = process.env.E2B_API_KEY || ''

// Remove this later, it's just for my debugging
console.log('MISTRAL_API_KEY:', MISTRAL_API_KEY ? 'Loaded' : 'Not Loaded')
console.log('E2B_API_KEY:', E2B_API_KEY ? 'Loaded' : 'Not Loaded')

const MODEL_NAME = 'codestral-latest'
const SYSTEM_PROMPT = `
You're a python data scientist that is analyzing historical mathematicians. You are given tasks to complete and you run Python code to solve them.

Information about the mathematicians dataset:
- It's in the \`/home/user/mathematicians.csv\` file
- The CSV file is using , as the delimiter
- It has the following columns (examples included):
    - Unnamed: 0: 0, 1, 2
    - name: "Asger Hartvig Aaboe", "Ernst Abbe", "Edwin Abbott Abbott"
    - gender: "M", "F"
    - born_date: "26 April 1922", "23 January 1840", "20 December 1838"
    - born_year: 1922, 1840, 1838
    - born_place: "Copenhagen, Denmark", "Eisenach, Grand Duchy of Saxe-Weimar-Eisenach (now in Germany)", "Marylebone, Middlesex, England"
    - born_country: "Denmark", "Germany", "United Kingdom"
    - died_date: "19 January 2007", "14 January 1905", "12 October 1926"
    - died_year: 2007, 1905, 1926
    - died_place: "North Haven, Connecticut, USA", "Jena, Germany", "Hampstead, London, England"
    - died_country: "United States", "Germany", "United Kingdom"
    - degree: "Ph.D.", "Dr. phil.", "Dr. rer. nat."
    - degree_generalized: "Ph.D.", "M.A.", "Other"
    - university: "Brown University", "Georg-August-Universität Göttingen", "Universitetet i Oslo"
    - university_country: "United States", "Germany", "Norway"
    - graduate_year: 1957, 1861, 1822
    - main_classification: 1, 14, 12
    - classification_indices: "['01']", "['01', '15']", "['14', '48', '00', '32', '09', '01', '69', '58', '55', '52', '50', '46', '20', '18', '16', '15', '04']"
    - earliest_publication: 1954, 1991, 1826
    - publication_count: 19, 7, 25
    - advisor_count: 1, 0, 2
    - student_count: 4, 2, 0
    - descendant_count: 5, 10, 0
    - summary: "Asger Aaboe was a Danish mathematician who is known for his contributions to the history of ancient Babylonian astronomy.", "Ernst Abbe was a German instrument maker who made important improvements in lens design.", "Edwin Abbott was an English schoolmaster who wrote the popular book Flatland as an introduction to higher dimensions."

Generally, you follow these rules:
- ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN
- ALWAYS RESPOND ONLY WITH CODE IN CODE BLOCK LIKE THIS:
\`\`\`python
{code}
\`\`\`
- the python code runs in jupyter notebook.
- every time you generate python, the code is executed in a separate cell. it's okay to multiple calls to \`execute_python\`.
- display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
- you have access to the internet and can make api requests.
- you also have access to the filesystem and can read/write files.
- you can install any pip package (if it exists) if you need to be running \`!pip install {package}\`. The usual packages for data analysis are already preinstalled though.
- you can run any python code you want, everything is running in a secure sandbox environment
`

const client = new MistralClient()

async function codeInterpret(codeInterpreter: CodeInterpreter, code: string): Promise<Result[]> {
    console.log('Running code interpreter...')

    const exec = await codeInterpreter.notebook.execCell(code, {
        onStderr: (msg: ProcessMessage) => console.log('[Code Interpreter stderr]', msg),
        onStdout: (stdout: ProcessMessage) => console.log('[Code Interpreter stdout]', stdout),
    })

    if (exec.error) {
        console.log('[Code Interpreter ERROR]', exec.error)
        throw new Error(exec.error.value)
    }

    return exec.results
}

async function chat(codeInterpreter: CodeInterpreter, userMessage: string): Promise<Result[]> {
    console.log(`\n${'='.repeat(50)}\nUser Message: ${userMessage}\n${'='.repeat(50)}`)

    const messages = [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userMessage }
    ]

    try {
        const response = await client.chat({
            model: MODEL_NAME,
            messages: messages,
        })

        const responseMessage = response.choices[0].message.content
        const codeBlockMatch = responseMessage.match(/```python\n([\s\S]*?)\n```/)

        if (codeBlockMatch && codeBlockMatch[1]) {
            const pythonCode = codeBlockMatch[1]
            console.log('CODE TO RUN')
            console.log(pythonCode)
            const codeInterpreterResults = await codeInterpret(codeInterpreter, pythonCode)
            return codeInterpreterResults
        } else {
            console.log('Failed to match any Python code in model\'s response')
            return []
        }
    } catch (error) {
        console.error('Error during API call:', error)
        throw error
    }
}

async function uploadDataset(codeInterpreter: CodeInterpreter): Promise<string> {
    console.log('Uploading dataset to Code Interpreter sandbox...')
    const datasetPath = './mathematicians.csv'

    if (!fs.existsSync(datasetPath)) {
        throw new Error('Dataset file not found')
    }

    // Read the file into a buffer
    const fileBuffer = fs.readFileSync(datasetPath)

    try {
        const remotePath = await codeInterpreter.uploadFile(fileBuffer, 'mathematicians.csv') // Pass the buffer and filename
        if (!remotePath) {
            throw new Error('Failed to upload dataset')
        }
        console.log('Uploaded at', remotePath)
        return remotePath
    } catch (error) {
        console.error('Error during file upload:', error)
        throw error
    }
}

async function run() {
    const codeInterpreter = await CodeInterpreter.create()

    try {
        const remotePath = await uploadDataset(codeInterpreter)
        console.log('Remote path of the uploaded dataset:', remotePath)

        const codeInterpreterResults = await chat(
            codeInterpreter,
            'Make a chart showing linear regression of the relationship between the year a mathematician was born and number of publications. Filter out any missing values or values in wrong format.'
            // 'Make a chart showing average life length of mathematicians in time.'
            // 'Make a doughnut chart showing the 20 most frequent universities where mathematicians studied.'
            // 'Make a pie chart showing 10 most frequent countries where mathematicians were born.'
            // 'Make a scatter plot chart showing showing number of mathematics' publications throughot the history'
            // 'Make a timeline showing points in time when Russian mathematicians were born. On the x axis is the time, the y axis is just constant 1 if a mathematician was born, 0 if not.'
            // 'Make a stacked bar chart showing for 5 random mathematicians the number of publications, and number of descendants they had.'
            // 'Make an area chart with two variables of your choice.'
            // 'Make a bubble chart with two variables of your choice.'
            // 'Make a chart showing dummy regression of the relationship between whether a mathematician has (1) or doesnt have (0) a degree and number of publications. Filter out any missing values or values in wrong format.'
        )
        console.log('codeInterpreterResults:', codeInterpreterResults)

        const result = codeInterpreterResults[0]
        console.log('Result object:', result)

        if (result && result.png) {
            fs.writeFileSync('image_1.png', Buffer.from(result.png, 'base64'))
            console.log('Success: Image generated and saved as image_1.png')
        } else {
            console.log('Error: No PNG data available.')
        }

    } catch (error) {
        console.error('An error occurred:', error)
    } finally {
        await codeInterpreter.close()
    }
}

run()
