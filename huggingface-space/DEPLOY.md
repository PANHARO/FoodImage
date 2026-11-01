# Upload to Hugging Face Spaces

The folder is ready to upload, including the 16.4 MB trained checkpoint and five example images. Nothing in these steps changes either GitHub source repository or the Canva presentation.

## 1. Create the Space

1. Sign in at [Hugging Face](https://huggingface.co/).
2. Choose **New Space** at [huggingface.co/new-space](https://huggingface.co/new-space).
3. Use a name such as `sea-food-recognition`, choose **Gradio**, **CPU Basic**, and **Public** for a publicly embedded portfolio demo.
4. Create the Space under your own account. Keep its exact account/name for the commands below.

## 2. Install Git LFS and sign in

Install [Git](https://git-scm.com/downloads) and [Git LFS](https://git-lfs.com/) if needed. Then, in PowerShell:

```powershell
git lfs install
python -m pip install --upgrade huggingface_hub
hf auth login --add-to-git-credential
```

Use a Hugging Face write token scoped to your Space when prompted. Do not paste the token into source code or the remote URL.

Hugging Face now uses Xet storage and recommends Git Xet for new Git workflows. Its [compatibility bridge supports existing Git LFS clients](https://huggingface.co/docs/hub/xet/legacy-git-lfs). The commands here use Git LFS as requested; the CLI upload alternative below handles current large-file storage automatically.

## 3. Clone the new Space and copy this package

Replace `YOUR_USERNAME` with your Hugging Face account name:

```powershell
git clone https://huggingface.co/spaces/YOUR_USERNAME/sea-food-recognition
cd sea-food-recognition
```

Copy these items from the extracted package into that cloned folder, keeping the directory structure:

```text
app.py
requirements.txt
README.md
DEPLOY.md
embed.html
.gitattributes
.gitignore
model/
  efficientnet-b0.pth
examples/
  adobo.jpg
  amok_trey.jpg
  banh_mi.jpg
  pho.jpg
  satay.jpg
  SOURCES.md
```

Copy the package's README over the generated template README. Merge the package's `*.pth` line into any existing `.gitattributes` so other generated tracking rules are preserved.

## 4. Track and push the weights with Git LFS

Run this before staging the weights:

```powershell
git lfs track "*.pth"
git add .gitattributes .gitignore app.py requirements.txt README.md DEPLOY.md embed.html examples model
git lfs ls-files
git commit -m "Add SEA food recognition portfolio demo"
git push origin main
```

`git lfs ls-files` must list `model/efficientnet-b0.pth`. If it does not, fix tracking before committing:

```powershell
git rm --cached -- model/efficientnet-b0.pth
git add .gitattributes model/efficientnet-b0.pth
git lfs ls-files
```

The `--cached` command only removes the staged copy; the local weights remain.

Alternatively, after authentication, upload the extracted folder directly:

```powershell
hf upload YOUR_USERNAME/sea-food-recognition . --repo-type space
```

The [Hugging Face CLI](https://huggingface.co/docs/hub/repositories-getting-started#adding-files-to-a-repository-cli) handles large files automatically. Do not upload the ZIP as a single file: `app.py` and `README.md` must be at the Space repository root.

## 5. Verify the build

1. Open the Space's **App** tab and wait for the build to finish.
2. Check the logs if it reports a runtime error. The package pins Python 3.12, Gradio 6.31.0, and paired CPU PyTorch wheels.
3. Click each of the five examples. Upload a separate JPG or PNG and check that it predicts automatically.
4. Clear the image and check that old scores disappear.
5. Low-confidence results clear the bars and show the complete ten-dish message. The cutoff is strictly `< 0.5`; exactly 50% is accepted.
6. Confirm the weights are actual binary data, not a Git LFS pointer. The file must be **16,380,921 bytes**; for a local clone, run `git lfs pull` if needed.

Optional local checksum:

```powershell
Get-FileHash model/efficientnet-b0.pth -Algorithm SHA256
```

Expected: `213f7d01da9716fef0e1268ff07b72f3ece4c14db09ae70a3adcad015edb48eb`.

## 6. Embed in the portfolio

1. In the running Space menu, choose **Embed this Space** and copy its exact `https://…hf.space` URL.
2. Open `embed.html` and replace **both** occurrences of `https://YOUR-SPACE-SUBDOMAIN.hf.space`.
3. Copy the section and its styles into the portfolio.
4. Keep the wake-up note: [free Spaces sleep when unused](https://huggingface.co/docs/hub/spaces-overview#lifecycle) and the first visit can take longer.
5. The note stays visible as context. An iframe's `load` event can fire on a waking screen, so it is not used to claim the model is ready.
6. Test at desktop and mobile widths. The iframe has extra mobile height and can scroll if needed.

See the [official iframe documentation](https://huggingface.co/docs/hub/spaces-embed) for embedding details.
