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
You're a python data scientist that is analyzing daily temperature of major cities. You are given tasks to complete and you run Python code to solve them.

Information about the temperature dataset:
- It's in the \`/home/user/city_temperature.csv\` file
- The CSV file is using \`,\` as the delimiter
- It has following columns (examples included):
  - \`Region\`: "North America", "Europe"
  - \`Country\`: "Iceland"
  - \`State\`: for example "Texas" but can also be null
  - \`City\`: "Prague"
  - \`Month\`: "June"
  - \`Day\`: 1-31
  - \`Year\`: 2002
  - \`AvgTemperature\`: temperature in Celsius, for example 24

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
    const datasetPath = './city_temperature.csv'

    if (!fs.existsSync(datasetPath)) {
        throw new Error('Dataset file not found')
    }

    // Read the file into a buffer
    const fileBuffer = fs.readFileSync(datasetPath)

    try {
        const remotePath = await codeInterpreter.uploadFile(fileBuffer, 'city_temperature.csv') // Pass the buffer and filename
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
            'Plot average temperature over the years in Algeria'
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
