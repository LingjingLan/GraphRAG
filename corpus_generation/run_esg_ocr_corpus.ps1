# ESG OCR corpus (PDF -> MD -> esg_ocr.json)
# Requires: docker compose stack with paddleocr + redis (backend mounts ESG_report / ESG_KG / Datasets)

cd $PSScriptRoot\..

$argsList = $args
if (-not $argsList -or $argsList.Count -eq 0) {
  $argsList = @("--limit", "1")
}

Write-Host "Recreate backend to pick up volume mounts (ESG_report, ESG_KG, Datasets)..."
docker compose up -d backend

Write-Host "Running: build_esg_ocr_corpus.py $($argsList -join ' ')"
docker exec -it euleresg-backend-dev python3 /workspace/backend/scripts/build_esg_ocr_corpus.py @argsList
