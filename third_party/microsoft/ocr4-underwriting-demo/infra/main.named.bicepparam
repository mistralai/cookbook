using './main.bicep'

// Canonical parameters for the deployed stack (Cloud Adoption Framework naming).
// This reproduces the live resources in rg-mistral-ocr-example.

// Globally-unique account name (becomes the endpoint subdomain).
param accountName = 'aif-mistral-docai'

// Must be a region that supports the Mistral models.
param location = 'westus'

// OCR model (GlobalStandard quota is exhausted on this subscription; DataZoneStandard has room).
param ocrDeploymentName = 'mistral-ocr-4'
param ocrSku = 'DataZoneStandard'
param ocrCapacity = 10

// Reasoning model. On, to back the three Foundry Agent Service agents on
// proj-mortgage-doc-pipeline (intake, underwriting, extractor). A fresh deployment on
// 2026-08-25 passed a 143-token probe with no cap, so the P0-2 ~62-token symptom did not
// recur here. The OCR-4-only app path does not use this model. Set deployMedium=false to
// remove it after the session.
param deployMedium = true
param mediumDeploymentName = 'mistral-medium-3-5'
param mediumSku = 'GlobalStandard'
param mediumCapacity = 10

// Foundry project + monitoring, named for purpose.
param projectName = 'proj-mortgage-doc-pipeline'
param logAnalyticsName = 'log-mistral-docai'
param appInsightsName = 'appi-mistral-docai'

// Storage account (globally unique, distinct prefix).
param storageAccountName = 'stmistraldocai0819'
