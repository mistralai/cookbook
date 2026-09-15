// Function App hosting (Flex Consumption) — runs the queue-triggered pipeline in Azure.
// Deployed separately from main.bicep; references the existing resources by name.
//
//   az deployment group create -g <rg> -f infra/functionapp.bicep \
//     -p foundryAccountName=mistral-ocr-foundry-mistralai \
//        storageAccountName=<storageAccountName> \
//        appInsightsName=mistral-ocr-foundry-mistralai-ai

@description('Location (Flex Consumption is available in westus/westus2/westus3, etc.).')
param location string = resourceGroup().location

@description('Foundry (AIServices) account name — source of the model key + endpoints.')
param foundryAccountName string

@description('Storage account name (blob + queue + deployment package).')
param storageAccountName string

@description('Application Insights component name.')
param appInsightsName string

@description('Function app name (globally unique).')
param functionAppName string = '${foundryAccountName}-func'

@description('OCR deployment name.')
param ocrDeploymentName string = 'mistral-ocr-4'

@description('Chat/reasoning deployment name.')
param chatDeploymentName string = 'mistral-medium-3-5'

@description('Foundry project endpoint. Enables the hosted-agent extractor path in the deployed Function. Empty string leaves the Function on the inline extractor.')
param projectEndpoint string = ''

@description('Python version for the Flex runtime.')
param pythonVersion string = '3.12'

resource storage 'Microsoft.Storage/storageAccounts@2024-01-01' existing = {
  name: storageAccountName
}
resource foundry 'Microsoft.CognitiveServices/accounts@2026-07-01' existing = {
  name: foundryAccountName
}
resource appInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: appInsightsName
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2024-01-01' existing = {
  parent: storage
  name: 'default'
}

// Container that holds the deployed app package (required by Flex Consumption).
resource deployContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2024-01-01' = {
  parent: blobService
  name: 'app-package'
}

var storageConn = 'DefaultEndpointsProtocol=https;AccountName=${storage.name};AccountKey=${storage.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
var foundryHost = 'https://${foundry.name}.services.ai.azure.com'

resource plan 'Microsoft.Web/serverfarms@2024-04-01' = {
  name: '${functionAppName}-plan'
  location: location
  sku: {
    name: 'FC1'
    tier: 'FlexConsumption'
  }
  kind: 'functionapp'
  properties: {
    reserved: true
  }
}

resource funcApp 'Microsoft.Web/sites@2024-04-01' = {
  name: functionAppName
  location: location
  kind: 'functionapp,linux'
  // azd matches the `api` service in azure.yaml to this resource by this tag, then publishes
  // the Function code to it. Without the tag, `azd deploy` fails with "unable to find a
  // resource tagged with 'azd-service-name: api'".
  tags: {
    'azd-service-name': 'api'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    // Queue-triggered app with no HTTP routes, but pin the baseline anyway: never serve plaintext
    // HTTP, and disable FTP publishing.
    httpsOnly: true
    serverFarmId: plan.id
    functionAppConfig: {
      deployment: {
        storage: {
          type: 'blobContainer'
          value: '${storage.properties.primaryEndpoints.blob}app-package'
          authentication: {
            type: 'StorageAccountConnectionString'
            storageAccountConnectionStringName: 'DEPLOYMENT_STORAGE_CONNECTION_STRING'
          }
        }
      }
      scaleAndConcurrency: {
        maximumInstanceCount: 40
        instanceMemoryMB: 2048
      }
      runtime: {
        name: 'python'
        version: pythonVersion
      }
    }
    siteConfig: {
      minTlsVersion: '1.2'
      ftpsState: 'Disabled'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: storageConn
        }
        {
          name: 'DEPLOYMENT_STORAGE_CONNECTION_STRING'
          value: storageConn
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
        {
          // Let the Python worker export its OpenTelemetry spans (the OCR HTTP dependency and the
          // pipeline's manual spans) to Application Insights. Works with host.json
          // telemetryMode=OpenTelemetry; without it the host owns telemetry and worker spans are
          // dropped, so the OCR call never appears in a trace.
          name: 'PYTHON_APPLICATIONINSIGHTS_ENABLE_TELEMETRY'
          value: 'true'
        }
        {
          name: 'AZURE_AI_KEY'
          value: foundry.listKeys().key1
        }
        {
          name: 'AZURE_OCR_ENDPOINT'
          value: '${foundryHost}/providers/mistral/azure/ocr'
        }
        {
          name: 'AZURE_OCR_DEPLOYMENT'
          value: ocrDeploymentName
        }
        {
          name: 'AZURE_INFERENCE_ENDPOINT'
          value: foundryHost
        }
        {
          name: 'AZURE_CHAT_DEPLOYMENT'
          value: chatDeploymentName
        }
        {
          name: 'AZURE_AI_PROJECT_ENDPOINT'
          value: projectEndpoint
        }
        {
          // Route the pipeline extractor through the hosted Foundry agent only when BOTH a project
          // endpoint AND a chat deployment exist. Gating on the endpoint alone turned agents on with
          // no backing model when deployMedium=false (chatDeploymentName=''), leaving the extractor
          // agent nothing to call; the OCR-annotation path still fills the fields in that case.
          name: 'USE_FOUNDRY_AGENTS'
          value: (empty(projectEndpoint) || empty(chatDeploymentName)) ? 'false' : 'true'
        }
        {
          name: 'AZURE_STORAGE_CONNECTION_STRING'
          value: storageConn
        }
        {
          name: 'AZURE_STORAGE_INBOX_CONTAINER'
          value: 'inbox'
        }
        {
          name: 'AZURE_STORAGE_RESULTS_CONTAINER'
          value: 'results'
        }
        {
          name: 'ENABLE_SENSITIVE_TELEMETRY'
          value: 'true'
        }
        {
          name: 'OTEL_SERVICE_NAME'
          value: 'doc-pipeline'
        }
      ]
    }
  }
}

// Grant the Function's managed identity data-plane access to the Foundry account, so its
// managed-identity token (scope https://ai.azure.com/.default) is accepted by the Agent
// Service agents API. Without this the hosted-agent path 401s and falls back to
// /chat/completions. Azure AI User grants the Microsoft.CognitiveServices/* data action plus
// reader on the account and its projects. Alternate role if this one is unavailable in a
// tenant: Cognitive Services User a97b65f3-24c7-4388-baec-2e87135dc908 (same data action).
// Creating this assignment requires the deployer to have Owner or User Access Administrator.
var azureAiUserRoleId = '53ca6127-db72-4b80-b1b0-d745d6d5456d' // Azure AI User

resource funcFoundryRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(foundry.id, funcApp.id, azureAiUserRoleId)
  scope: foundry
  properties: {
    principalId: funcApp.identity.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', azureAiUserRoleId)
    principalType: 'ServicePrincipal'
  }
}

output functionAppName string = funcApp.name
output functionAppHostName string = funcApp.properties.defaultHostName
