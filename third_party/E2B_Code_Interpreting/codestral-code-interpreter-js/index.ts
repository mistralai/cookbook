import fs from 'node:fs'
import { CodeInterpreter, Result, ProcessMessage } from '@e2b/code-interpreter'
import * as dotenv from 'dotenv'
import MistralClient from '@mistralai/mistralai'

dotenv.config()

const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY || ''
const E2B_API_KEY = process.env.E2B_API_KEY || ''

if (!MISTRAL_API_KEY) {
    console.error('Error: MISTRAL_API_KEY is not provided. Please set the MISTRAL_API_KEY in your environment variables.')
    process.exit(1)
}

if (!E2B_API_KEY) {
    console.error('Error: E2B_API_KEY is not provided. Please set the E2B_API_KEY in your environment variables.')
    process.exit(1)
}

console.log('MISTRAL_API_KEY:', MISTRAL_API_KEY ? 'Loaded' : 'Not Loaded')
console.log('E2B_API_KEY:', E2B_API_KEY ? 'Loaded' : 'Not Loaded')

const MODEL_NAME = 'codestral-latest'
const SYSTEM_PROMPT = `
You're a python data scientist. You are given tasks to complete and you run Python code to solve them.

Information about the csv dataset:
- It's in the \`/home/user/global_economy_indicators.csv\` file
- The CSV file is using , as the delimiter
- It has the following columns (examples included):
    - country: "Argentina", "Australia"
    - Region: "SouthAmerica", "Oceania"
    - Surface area (km2): for example, 2780400
    - Population in thousands (2017): for example, 44271
    - Population density (per km2, 2017): for example, 16.2
    - Sex ratio (m per 100 f, 2017): for example, 95.9
    - GDP: Gross domestic product (million current US$): for example, 632343
    - GDP growth rate (annual %, const. 2005 prices): for example, 2.4
    - GDP per capita (current US$): for example, 14564.5
    - Economy: Agriculture (% of GVA): for example, 10.0
    - Economy: Industry (% of GVA): for example, 28.1
    - Economy: Services and other activity (% of GVA): for example, 61.9
    - Employment: Agriculture (% of employed): for example, 4.8
    - Employment: Industry (% of employed): for example, 20.6
    - Employment: Services (% of employed): for example, 74.7
    - Unemployment (% of labour force): for example, 8.5
    - Employment: Female (% of employed): for example, 43.7
    - Employment: Male (% of employed): for example, 56.3
    - Labour force participation (female %): for example, 48.5
    - Labour force participation (male %): for example, 71.1
    - International trade: Imports (million US$): for example, 59253
    - International trade: Exports (million US$): for example, 57802
    - International trade: Balance (million US$): for example, -1451
    - Education: Government expenditure (% of GDP): for example, 5.3
    - Health: Total expenditure (% of GDP): for example, 8.1
    - Health: Government expenditure (% of total health expenditure): for example, 69.2
    - Health: Private expenditure (% of total health expenditure): for example, 30.8
    - Health: Out-of-pocket expenditure (% of total health expenditure): for example, 20.2
    - Health: External health expenditure (% of total health expenditure): for example, 0.2
    - Education: Primary gross enrollment ratio (f/m per 100 pop): for example, 111.5/107.6
    - Education: Secondary gross enrollment ratio (f/m per 100 pop): for example, 104.7/98.9
    - Education: Tertiary gross enrollment ratio (f/m per 100 pop): for example, 90.5/72.3
    - Education: Mean years of schooling (female): for example, 10.4
    - Education: Mean years of schooling (male): for example, 9.7
    - Urban population (% of total population): for example, 91.7
    - Population growth rate (annual %): for example, 0.9
    - Fertility rate (births per woman): for example, 2.3
    - Infant mortality rate (per 1,000 live births): for example, 8.9
    - Life expectancy at birth, female (years): for example, 79.7
    - Life expectancy at birth, male (years): for example, 72.9
    - Life expectancy at birth, total (years): for example, 76.4
    - Military expenditure (% of GDP): for example, 0.9
    - Population, female: for example, 22572521
    - Population, male: for example, 21472290
    - Tax revenue (% of GDP): for example, 11.0
    - Taxes on income, profits and capital gains (% of revenue): for example, 12.9
    - Urban population (% of total population): for example, 91.7

Generally, you follow these rules:
- ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN
- ALWAYS RESPOND ONLY WITH CODE IN CODE BLOCK LIKE THIS:
\`\`\`python
{code}
\`\`\`
- the Python code runs in jupyter notebook.
- every time you generate Python, the code is executed in a separate cell. it's okay to make multiple calls to \`execute_python\`.
- display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
- you have access to the internet and can make api requests.
- you also have access to the filesystem and can read/write files.
- you can install any pip package (if it exists) if you need to be running \`!pip install {package}\`. The usual packages for data analysis are already preinstalled though.
- you can run any Python code you want, everything is running in a secure sandbox environment
`

const client = new MistralClient()

async function codeInterpret(codeInterpreter: CodeInterpreter, code: string): Promise<Result[]> {
    console.log('Running code interpreter...')

    const exec = await codeInterpreter.notebook.execCell(code, {
        onStderr: (msg: ProcessMessage) => console.log('[Code Interpreter stderr]', msg),
        onStdout: (stdout: ProcessMessage) => console.log('[Code Interpreter stdout]', stdout)
    })

    if (exec.error) {
        console.error('[Code Interpreter ERROR]', exec.error)
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
            messages: messages
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
            console.error('Failed to match any Python code in model\'s response')
            return []
        }
    } catch (error) {
        console.error('Error during API call:', error)
        throw error
    }
}

async function uploadDataset(codeInterpreter: CodeInterpreter): Promise<string> {
    console.log('Uploading dataset to Code Interpreter sandbox...')
    const datasetPath = './global_economy_indicators.csv'

    if (!fs.existsSync(datasetPath)) {
        throw new Error('Dataset file not found')
    }

    // Read the file into a buffer
    const fileBuffer = fs.readFileSync(datasetPath)

    try {
        const remotePath = await codeInterpreter.uploadFile(fileBuffer, 'global_economy_indicators.csv') // Pass the buffer and filename
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
            // Task for the model
            'Make a chart showing linear regression of the relationship between GDP per capita and life expectancy from the global_economy_indicators. Filter out any missing values or values in wrong format.'
        )
        console.log('codeInterpreterResults:', codeInterpreterResults)

        const result = codeInterpreterResults[0]
        console.log('Result object:', result)

        if (result && result.png) {
            fs.writeFileSync('image_1.png', Buffer.from(result.png, 'base64'))
            console.log('Success: Image generated and saved as image_1.png')
        } else {
            console.error('Error: No PNG data available.')
        }

    } catch (error) {
        console.error('An error occurred:', error)
    } finally {
        await codeInterpreter.close()
    }
}

run()
