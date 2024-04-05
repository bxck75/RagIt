git config --global user.email "goldenkooy@gmail.com"
git config --global user.name "bxck75"

echo "# RagIt" >> README.md
git init
git add README.md
git add .
git commit -m "add files"
git branch -M main
git remote add origin https://github.com/bxck75/RagIt.git
git push -u origin main