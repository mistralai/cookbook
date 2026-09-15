// Observability alerts (scheduled-query rules over Application Insights).
// Deployed separately from main.bicep so an alert-schema issue can't block core infra.
//
//   az deployment group create -g <rg> -f infra/alerts.bicep \
//     -p appInsightsId=<appInsightsId output from main> location=westus

@description('Application Insights resource id (from main.bicep output appInsightsId).')
param appInsightsId string

@description('Location for the alert rules.')
param location string = resourceGroup().location

// Fires when any exception is recorded by the pipeline / agents.
resource exceptionsAlert 'Microsoft.Insights/scheduledQueryRules@2023-03-15-preview' = {
  name: 'pipeline-exceptions'
  location: location
  properties: {
    displayName: 'Pipeline exceptions detected'
    description: 'Any exception recorded in App Insights over the last 15 minutes.'
    severity: 2
    enabled: true
    scopes: [
      appInsightsId
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      allOf: [
        {
          query: 'exceptions'
          timeAggregation: 'Count'
          operator: 'GreaterThan'
          threshold: 0
          failingPeriods: {
            numberOfEvaluationPeriods: 1
            minFailingPeriodsToAlert: 1
          }
        }
      ]
    }
    autoMitigate: true
    actions: {}
  }
}

// Fires when OCR/model calls are throttled (HTTP 429 on outbound dependencies).
resource throttlingAlert 'Microsoft.Insights/scheduledQueryRules@2023-03-15-preview' = {
  name: 'model-throttling-429'
  location: location
  properties: {
    displayName: 'Model calls throttled (429)'
    description: 'Outbound OCR/model dependency calls returning HTTP 429 over the last 15 minutes.'
    severity: 3
    enabled: true
    scopes: [
      appInsightsId
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      allOf: [
        {
          query: 'dependencies | where resultCode == "429"'
          timeAggregation: 'Count'
          operator: 'GreaterThan'
          threshold: 0
          failingPeriods: {
            numberOfEvaluationPeriods: 1
            minFailingPeriodsToAlert: 1
          }
        }
      ]
    }
    autoMitigate: true
    actions: {}
  }
}
