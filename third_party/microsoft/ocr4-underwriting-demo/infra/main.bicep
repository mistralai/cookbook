// Mistral OCR 4 on Azure Foundry: infrastructure (Medium 3.5 optional, off by default)
//
// Provisions:
//   1. A Microsoft Foundry (AIServices) account with project management enabled
//   2. A Foundry project, plus an Application Insights connection on it, so observability
//      (Tracing, Monitoring) surfaces in the Foundry portal, not only the Azure portal
//   3. A deployment of mistral-ocr-4-0   (document OCR)
//   4. Optionally, a deployment of mistral-medium-3-5 (off by default; OCR-4-only pipeline)
//   5. A Storage account with `inbox` (uploads) + `results` containers
//
// The project and connection are additive: deploying this template to the existing
// resource group is an incremental update, not a teardown of the account or the models.
//
// Deploy:
//   az deployment group create -g <rg> -f infra/main.bicep -p infra/main.named.bicepparam

@description('Optional salt appended to the derived account name so a poisoned name (soft-deleted and unpurgeable) can be rotated without changing the resource group. Empty keeps the original deterministic name byte-for-byte. Set it via AZURE_FOUNDRY_NAME_SUFFIX in the azd path (scripts/up.sh --new-name does this for you). Ignored when accountName is set explicitly, as in main.named.bicepparam.')
param nameSuffix string = ''

@description('Name of the Foundry (AIServices) account. Globally unique, becomes the endpoint subdomain. Defaults to a name derived from the resource group so an unattended azd up never stalls; main.named.bicepparam overrides it for the manual named stack. The default is deterministic per RG, so re-applies are idempotent. One trap: teardown soft-deletes the account AND a backing Azure ML workspace shadow of the same name (Kind: AmlRp). That shadow purges asynchronously, never appears in the Cognitive Services soft-deleted list, and on some subscriptions has no queryable purge endpoint, so a same-RG, same-name redeploy can fail with "Soft-deleted workspace exists". scripts/down.sh fires a best-effort purge of the shadow; where that endpoint is unavailable, set a fresh nameSuffix to rotate onto a clean name. (Foundry projects do not soft-delete, so there is no separate project to purge.)')
param accountName string = empty(nameSuffix) ? 'aif${uniqueString(resourceGroup().id)}' : 'aif${uniqueString(resourceGroup().id, nameSuffix)}'

@description('Azure region. Must support the Mistral models (e.g. westus, eastus, eastus2, westus3, southcentralus, northcentralus, swedencentral).')
param location string = resourceGroup().location

// ---- OCR model ----
@description('OCR deployment name — goes in the `model` field of OCR calls (NOT the catalog name).')
param ocrDeploymentName string = 'mistral-ocr-4'

@description('OCR deployment SKU. OCR 4 requires GlobalStandard or DataZoneStandard.')
@allowed([
  'GlobalStandard'
  'DataZoneStandard'
])
param ocrSku string = 'DataZoneStandard'

@description('OCR provisioned capacity (requests per minute units). Set to 10 so the demo, which uploads several documents in quick succession, does not hit 429 throttling on OCR.')
param ocrCapacity int = 10

// ---- Reasoning model (optional, off by default) ----
// The pipeline runs OCR-4-only: field extraction comes from OCR 4's custom annotation, so no
// chat model is required. Fresh Medium 3.5 deployments currently serve a broken ~62-token
// context, so this stays off unless that platform issue clears. Set deployMedium to true to
// add the model back.
@description('Deploy the Medium 3.5 reasoning model. Off by default (OCR-4-only pipeline).')
param deployMedium bool = false

@description('Medium 3.5 deployment name, used only when deployMedium is true.')
param mediumDeploymentName string = 'mistral-medium-3-5'

@description('Medium 3.5 deployment SKU.')
@allowed([
  'GlobalStandard'
  'DataZoneStandard'
])
param mediumSku string = 'GlobalStandard'

@description('Medium 3.5 provisioned capacity (thousands of tokens per minute). Set to 50: the demo fans several agent calls (extractor, underwriter, doc-chat) over the same deployment in quick succession, and a large document context tips a capacity of 10 into 429 throttling.')
param mediumCapacity int = 50

// ---- Storage ----
@description('Storage account name (3-24 lowercase alphanumerics, globally unique).')
param storageAccountName string = 'docpipe${uniqueString(resourceGroup().id)}'

resource account 'Microsoft.CognitiveServices/accounts@2026-07-01' = {
  name: accountName
  location: location
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    // Enables the new Microsoft Foundry project experience
    allowProjectManagement: true
    // Required to expose the *.services.ai.azure.com endpoint used by the model routes
    customSubDomainName: accountName
    publicNetworkAccess: 'Enabled'
  }
}

resource ocrDeployment 'Microsoft.CognitiveServices/accounts/deployments@2026-07-01' = {
  parent: account
  name: ocrDeploymentName
  sku: {
    name: ocrSku
    capacity: ocrCapacity
  }
  properties: {
    model: {
      format: 'Mistral AI'
      name: 'mistral-ocr-4-0'
      version: '1'
    }
  }
}

// Cognitive Services serializes deployment operations on an account, so this depends on the
// OCR deployment to create them in sequence, not in parallel. Created only when deployMedium
// is true (the pipeline is OCR-4-only by default).
resource mediumDeployment 'Microsoft.CognitiveServices/accounts/deployments@2026-07-01' = if (deployMedium) {
  parent: account
  name: mediumDeploymentName
  dependsOn: [
    ocrDeployment
  ]
  sku: {
    name: mediumSku
    capacity: mediumCapacity
  }
  properties: {
    model: {
      format: 'Mistral AI'
      name: 'mistral-medium-3-5'
      version: '1'
    }
  }
}

resource storage 'Microsoft.Storage/storageAccounts@2024-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2024-01-01' = {
  parent: storage
  name: 'default'
}

resource inboxContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2024-01-01' = {
  parent: blobService
  name: 'inbox'
}

resource resultsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2024-01-01' = {
  parent: blobService
  name: 'results'
}

// Dead-letter sink for undeliverable Event Grid events.
resource deadletterContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2024-01-01' = {
  parent: blobService
  name: 'deadletter'
}

// ---- SQS-equivalent: Storage Queue + Event Grid bridge (Blob -> Queue) ----
@description('Name of the ingestion queue (SQS-equivalent) the Function is triggered by.')
param ingestQueueName string = 'ingest'

resource queueService 'Microsoft.Storage/storageAccounts/queueServices@2024-01-01' = {
  parent: storage
  name: 'default'
}

resource ingestQueue 'Microsoft.Storage/storageAccounts/queueServices/queues@2024-01-01' = {
  parent: queueService
  name: ingestQueueName
}

// Event Grid system topic on the storage account (source of BlobCreated events).
resource egTopic 'Microsoft.EventGrid/systemTopics@2025-02-15' = {
  name: '${storageAccountName}-egt'
  location: location
  properties: {
    source: storage.id
    topicType: 'Microsoft.Storage.StorageAccounts'
  }
}

// Route BlobCreated events on the `inbox` container into the ingest queue.
resource egSubscription 'Microsoft.EventGrid/systemTopics/eventSubscriptions@2025-02-15' = {
  parent: egTopic
  name: 'inbox-to-queue'
  properties: {
    destination: {
      endpointType: 'StorageQueue'
      properties: {
        resourceId: storage.id
        queueName: ingestQueueName
        queueMessageTimeToLiveInSeconds: 604800
      }
    }
    filter: {
      includedEventTypes: [
        'Microsoft.Storage.BlobCreated'
      ]
      subjectBeginsWith: '/blobServices/default/containers/inbox/'
    }
    eventDeliverySchema: 'EventGridSchema'
    deadLetterDestination: {
      endpointType: 'StorageBlob'
      properties: {
        resourceId: storage.id
        blobContainerName: 'deadletter'
      }
    }
  }
  dependsOn: [
    ingestQueue
    deadletterContainer
  ]
}

// ---- Observability: Log Analytics + App Insights + diagnostic settings ----
@description('Log Analytics workspace name.')
param logAnalyticsName string = '${accountName}-law'

@description('Application Insights component name.')
param appInsightsName string = '${accountName}-ai'

@description('Foundry project name (child of the account). Holds the Foundry portal observability views.')
param projectName string = '${accountName}-proj'

resource law 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: law.id
  }
}

// ---- Foundry project + Application Insights connection ----
// Foundry portal observability (Tracing, Monitoring, Evaluation) is scoped to a project and
// reads telemetry from the App Insights resource connected to that project. Creating the
// project and this connection is what surfaces our OpenTelemetry traces in the Foundry portal
// (ai.azure.com), not only the Azure portal. App Insights stays the backing store; nothing
// about the existing telemetry changes.
resource project 'Microsoft.CognitiveServices/accounts/projects@2025-04-01-preview' = {
  parent: account
  name: projectName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    displayName: 'Mistral OCR pipeline'
    description: 'Document processing pipeline: Mistral OCR 4 with custom annotation extraction.'
  }
}

// The connection is what the project Tracing and Monitoring tabs read from. The App Insights
// connection string is stored as the credential (authType ApiKey), matching how the Foundry
// portal wires it under Project details, Connected resources.
resource appInsightsConnection 'Microsoft.CognitiveServices/accounts/projects/connections@2025-04-01-preview' = {
  parent: project
  name: 'appinsights'
  properties: {
    category: 'AppInsights'
    target: appInsights.id
    authType: 'ApiKey'
    isSharedToAll: true
    credentials: {
      key: appInsights.properties.ConnectionString
    }
    metadata: {
      ApiType: 'Azure'
      ResourceId: appInsights.id
    }
  }
}

// Storage connection so the inbox/results account shows under the project's Connected
// resources in the Foundry portal. Account-key auth keeps it consistent with the key-based
// posture elsewhere and avoids a role assignment (which needs Owner/User Access Admin).
resource storageConnection 'Microsoft.CognitiveServices/accounts/projects/connections@2025-04-01-preview' = {
  parent: project
  name: 'storage-doc-inbox'
  properties: {
    category: 'AzureBlob'
    target: storage.properties.primaryEndpoints.blob
    authType: 'AccountKey'
    isSharedToAll: true
    credentials: {
      key: storage.listKeys().keys[0].value
    }
    metadata: {
      ApiType: 'Azure'
      ResourceId: storage.id
      AccountName: storage.name
      ContainerName: 'inbox'
    }
  }
}

// Foundry account: request/response, usage, audit logs + all metrics.
resource foundryDiag 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  scope: account
  name: 'to-law'
  properties: {
    workspaceId: law.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

// Storage blob + queue services: transactions, capacity, throttling.
resource blobDiag 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  scope: blobService
  name: 'to-law'
  properties: {
    workspaceId: law.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'Transaction'
        enabled: true
      }
    ]
  }
}

resource queueDiag 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  scope: queueService
  name: 'to-law'
  properties: {
    workspaceId: law.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'Transaction'
        enabled: true
      }
    ]
  }
}

// Event Grid system topic: delivery + publish failures.
resource egDiag 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  scope: egTopic
  name: 'to-law'
  properties: {
    workspaceId: law.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

// ---- Outputs (wire these into .env) ----
@description('Base Foundry endpoint (Azure AI inference lives under /models).')
output foundryEndpoint string = 'https://${account.name}.services.ai.azure.com'

@description('OCR route — set AZURE_OCR_ENDPOINT to this.')
output ocrEndpoint string = 'https://${account.name}.services.ai.azure.com/providers/mistral/azure/ocr'

@description('OCR deployment name — set AZURE_OCR_DEPLOYMENT to this.')
output ocrDeploymentName string = ocrDeployment.name

@description('Medium 3.5 deployment name (empty unless deployMedium is true); set AZURE_CHAT_DEPLOYMENT to this when used.')
output mediumDeploymentName string = deployMedium ? mediumDeployment.name : ''

@description('Foundry account name. Key: az cognitiveservices account keys list --name <accountName> -g <rg> --query key1 -o tsv')
output accountName string = account.name

@description('Storage account name (for blob upload + Function trigger).')
output storageAccountName string = storage.name

@description('Ingestion queue name (SQS-equivalent) the Function is triggered by.')
output ingestQueueName string = ingestQueue.name

@description('Application Insights connection string — set APPLICATIONINSIGHTS_CONNECTION_STRING to this.')
output appInsightsConnectionString string = appInsights.properties.ConnectionString

@description('Log Analytics workspace resource id (alerts + workbook scope).')
output logAnalyticsWorkspaceId string = law.id

@description('Application Insights resource id (scheduled-query alert scope).')
output appInsightsId string = appInsights.id

@description('Foundry project name (holds the portal observability views).')
output projectName string = project.name

@description('Foundry project endpoint. Set AZURE_AI_PROJECT_ENDPOINT to this (evaluation + Projects SDK).')
output projectEndpoint string = 'https://${account.name}.services.ai.azure.com/api/projects/${project.name}'

// ---- Function App host (module) ----
// Folded in so a single `azd up` provision pass creates the platform resources AND the
// Function host together. The manual two-step (deploy main then functionapp) still works
// because functionapp.bicep also stands alone with `existing` references.
@description('Deploy the Function App host in this provision pass. azd sets this true; the named manual stack can leave it false and deploy functionapp.bicep separately.')
param deployFunctionApp bool = true

module functionApp 'functionapp.bicep' = if (deployFunctionApp) {
  name: 'functionApp'
  params: {
    location: location
    foundryAccountName: account.name
    storageAccountName: storage.name
    appInsightsName: appInsights.name
    ocrDeploymentName: ocrDeployment.name
    chatDeploymentName: deployMedium ? mediumDeployment.name : ''
    projectEndpoint: 'https://${account.name}.services.ai.azure.com/api/projects/${project.name}'
  }
}

// ---- azd-named outputs (azd writes each Bicep output into the environment / hook env) ----
@description('Foundry (AIServices) account name.')
output foundryAccountName string = account.name

@description('Application Insights component name.')
output appInsightsName string = appInsights.name

@description('Chat/reasoning deployment name for hosted-agent registration + Function settings.')
output AZURE_CHAT_DEPLOYMENT string = deployMedium ? mediumDeployment.name : ''

@description('Foundry project endpoint, named for the azd hook env (agent registration reads this).')
output AZURE_AI_PROJECT_ENDPOINT string = 'https://${account.name}.services.ai.azure.com/api/projects/${project.name}'

@description('Function App name (empty when deployFunctionApp is false).')
output functionAppName string = functionApp.?outputs.functionAppName ?? ''
