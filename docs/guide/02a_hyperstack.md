# PART 2A - Hyperstack GPU machine + SSH + VS Code + WinSCP (connection from zero)

## BEFORE YOU START: what you are building in this part
Your laptop has no strong GPU, so you rent one. Hyperstack is a cloud company that rents computers with NVIDIA GPUs by the hour. You will: (1) install two programs on your laptop (VS Code and WinSCP), (2) make an SSH key (a digital key pair that lets your laptop log in securely), (3) rent a GPU computer ("VM" = virtual machine) on Hyperstack, and (4) connect to it three ways: PowerShell (a terminal), VS Code (an editor that works ON the VM) and WinSCP (drag-and-drop files).
| Word | Plain meaning |
| VM (virtual machine) | a computer in a data centre that you control over the internet |
| SSH | the secure way to log in to a Linux computer remotely; it uses port 22 |
| SSH key pair | two files: the private key (stays on your laptop, never share) and the public key (.pub, you give it to Hyperstack) |
| public IP | the internet address of your VM, for example 38.80.123.45 |
| environment (Hyperstack) | a folder in Hyperstack that holds your VMs and keys in one region |
| flavor (Hyperstack) | the hardware size: which GPU, how many CPUs and how much memory |
| image (OS image) | the operating system installed on the VM; ours already has the NVIDIA driver and Docker |
Result at the end of this part: VS Code on your laptop shows a terminal that says ubuntu@medscriptai-gpu and nvidia-smi prints "NVIDIA A100 80GB PCIe".

## SECTION 2A.1 - Install VS Code on your laptop
WHERE: web browser on your Windows laptop
GOAL: VS Code installed.
Step 1. Open https://code.visualstudio.com/download
Step 2. Click the Windows button (User Installer, x64).
Step 3. Open the downloaded file VSCodeUserSetup-x64-....exe.
Step 4. Click I accept the agreement, then Next.
Step 5. Keep the install folder, click Next, then Next again.
Step 6. On "Select Additional Tasks" tick Add "Open with Code" action to Windows Explorer file context menu and Add to PATH.
Step 7. Click Next, then Install, then Finish.
You should see: VS Code opens with a Welcome tab.
CHECKPOINT: VS Code is open.

## SECTION 2A.2 - Install the Remote - SSH extension in VS Code
WHERE: VS Code on your laptop
GOAL: VS Code can open folders on the GPU computer.
Step 1. Press Ctrl+Shift+X (Extensions panel opens on the left).
Step 2. In the search box type Remote - SSH.
Step 3. Click the result named Remote - SSH whose publisher is Microsoft.
Step 4. Click Install.
You should see: a new green-blue "><" icon in the bottom-left corner of VS Code.
Watch out: install the one from Microsoft only; ignore look-alikes.
CHECKPOINT: the extension shows "Installed" (or a gear icon instead of the Install button).

## SECTION 2A.3 - Hugging Face token and Llama licence
WHERE: web browser
GOAL: a Write token (password for scripts) and access to Llama 3.1.
Step 1. Open https://huggingface.co/join and create an account (skip if you have one).
Step 2. Open https://huggingface.co/settings/tokens
Step 3. Click Create new token.
Step 4. Click the Write tab.
Step 5. In Token name type medscriptai
Step 6. Click Create token.
Step 7. Click Copy and paste the token into a private Notepad file named secrets.txt on your laptop. It starts with hf_ and is shown only once.
Step 8. Open https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
Step 9. Fill in the licence form at the top and click Submit.
You should see: "Your request to access this repo has been submitted" (approval later shows "You have been granted access").
Watch out: never paste the token into code files or chats. You will store it on the VM in Section 2B.4.
CHECKPOINT: secrets.txt contains your hf_ token; the licence request is submitted.

## SECTION 2A.4 - Hyperstack account and credit
WHERE: web browser
GOAL: a Hyperstack account with money on it.
Step 1. Open https://authui.hyperstack.cloud/register
Step 2. Create the account (e-mail, password) and confirm your e-mail.
Step 3. Open https://console.hyperstack.cloud/billing/activation and activate billing.
Step 4. Open https://console.hyperstack.cloud/billing
Step 5. Add credit: US$30-40 is enough for the whole manual.
You should see: your credit balance on the Billing page.
Watch out: Hyperstack warns about a fake look-alike website; only use addresses ending in hyperstack.cloud [1].
CHECKPOINT: the Billing page shows your credit.

## SECTION 2A.5 - Create the SSH key on your laptop
WHERE: Windows laptop > Start menu > type PowerShell > click Windows PowerShell
GOAL: a private key file and a public key file in C:\Users\YOUR_NAME\.ssh
Step 1. Check that SSH is installed. Type and press Enter:
```
ssh -V
```
You should see: OpenSSH_for_Windows_9.x ...
Step 2. Create the key pair. Type and press Enter:
```
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\hyperstack_ed25519 -C "medscriptai"
```
Step 3. When it asks "Enter passphrase (empty for no passphrase)", press Enter.
Step 4. When it asks "Enter same passphrase again", press Enter.
You should see: "Your identification has been saved in C:\Users\YOUR_NAME\.ssh\hyperstack_ed25519" and a small square picture of random characters.
Step 5. Print the PUBLIC key:
```
Get-Content $env:USERPROFILE\.ssh\hyperstack_ed25519.pub
```
You should see: one line starting with ssh-ed25519 and ending with medscriptai
Step 6. Select that whole line with the mouse and press Ctrl+C (copy). Keep PowerShell open.
| Placeholder | Replace with | Example |
| YOUR_NAME | your Windows user folder name (shown in the "saved in" line) | rahul |
Watch out: the file WITHOUT .pub is your private key: never upload, e-mail or share it.
CHECKPOINT: the folder C:\Users\YOUR_NAME\.ssh contains hyperstack_ed25519 and hyperstack_ed25519.pub
If not: if "ssh is not recognized", open Settings > System > Optional features > View features > search OpenSSH Client > Install, close and reopen PowerShell, repeat Step 1.

## SECTION 2A.6 - Create an environment in Hyperstack
WHERE: web browser > https://console.hyperstack.cloud
GOAL: an environment named medscriptai-env (holds your VM and key).
Step 1. In the left menu click Virtual Machines.
Step 2. Click Deploy New Virtual Machine (the deploy form opens).
Step 3. Scroll to the Environment section.
Step 4. Click Create New Environment.
Step 5. In the name field type medscriptai-env
Step 6. Keep the suggested region and click Create (or Save).
You should see: medscriptai-env selected in the Environment section.
Watch out: the VM and the SSH key must be in the SAME environment, or the key cannot be used.
CHECKPOINT: medscriptai-env exists and is selected. Keep this form open for the next section.

## SECTION 2A.7 - Import your SSH key into Hyperstack
WHERE: Hyperstack console > the same Deploy New Virtual Machine form
GOAL: Hyperstack knows your public key.
Step 1. In the Environment / SSH key area click Create New SSH Key.
Step 2. Click the Import SSH key tab.
Step 3. In Environment select medscriptai-env
Step 4. Click in the public key box and press Ctrl+V (the ssh-ed25519 ... line from Section 2A.5).
Step 5. In Name type laptop-ed25519
Step 6. Click Import.
You should see: laptop-ed25519 selected as the SSH key in the form.
CHECKPOINT: laptop-ed25519 is listed for medscriptai-env.
If not: "invalid key" = you pasted the private key or only part of the line; repeat Section 2A.5 Step 5-6.

## SECTION 2A.8 - Deploy the GPU machine (every field)
WHERE: Hyperstack console > Deploy New Virtual Machine form
GOAL: one GPU computer that is starting up.
Step 1. Fill the form one field at a time, exactly as in this table.
| Field (in the form) | Value | Why |
| Flavor (GPU card) | A100-80G-PCIe, GPU count 1 (if sold out: L40, count 1) | 80 GB lets the teacher (14B) and your model (8B) run together in Parts 5-6 |
| Environment | medscriptai-env | holds your key |
| OS image (tab OS Images) | Server 24.04 LTS R570 CUDA 12.8 with Docker | NVIDIA driver 570 + CUDA 12.8 + Docker already installed [4] |
| SSH key | laptop-ed25519 | lets you log in |
| SSH Access | ON (toggle enabled) | opens port 22 in the firewall |
| Public IP Address | ON (toggle enabled) | you connect over the internet |
| Number of VMs to deploy | 1 | one machine |
| Virtual Machine Name (Additional settings) | medscriptai-gpu | easy to recognise |
| Jupyter Notebook, Cloud-init Script, Labels, Firewall Rules | leave empty / off | not needed |
Step 2. Look at Final Configuration and Price Breakdown and check the hourly price (about US$1.35 per hour for 1x A100 PCIe, verify).
Step 3. Click Deploy.
You should see: the Virtual Machines page with medscriptai-gpu and a status such as CREATING or BUILD.
Watch out: billing starts when the status becomes ACTIVE [2].
| If you see... | It means... | Do this next |
| A100-80G-PCIe is greyed out | sold out in this region | choose 1x L40, or create a second environment in another region and import the key there too |
| the image "Server 24.04 LTS R570 CUDA 12.8 with Docker" is missing | different images in this region | choose "Server 22.04 LTS R570 CUDA 12.8 with Docker" (same driver) |
| only "Server 24.04 LTS R595 CUDA 13.2 with Docker" fits | newer driver | use it; later set VLLM_IMAGE=vllm/vllm-openai:v0.32.0 and VLLM_ENABLE_CUDA_COMPATIBILITY=0 (Part 5) |
CHECKPOINT: medscriptai-gpu appears in the Virtual Machines list.

## SECTION 2A.9 - Wait for ACTIVE, copy the IP, check the SSH firewall
WHERE: Hyperstack console > Virtual Machines
GOAL: the VM is running and reachable on port 22.
Step 1. Wait until the Status column shows ACTIVE (usually 2-10 minutes; refresh the page).
Step 2. In the PUBLIC IP column, copy the address and paste it into secrets.txt as VM_IP = ...
Step 3. Click the VM name medscriptai-gpu (the VM details page opens).
Step 4. Click the Firewall tab.
Step 5. Make sure Enable SSH access is ON.
Step 6. Under Inbound Rules, check there is a TCP port 22 rule with status SUCCESS.
Step 7. Click the Networking tab and find Connect over SSH: it shows a command like ssh ubuntu@38.80.123.45 (you will add your key to it in the next section).
You should see: ACTIVE, a public IP, and the port 22 rule.
CHECKPOINT: you have VM_IP and the port 22 rule shows SUCCESS.
If not: no public IP = Networking tab > turn on Public IP Attached.

## SECTION 2A.10 - First connection from PowerShell
WHERE: Windows PowerShell on your laptop
GOAL: log in to the GPU computer and see the GPU.
Step 1. Type (replace VM_IP) and press Enter:
```
ssh -i $env:USERPROFILE\.ssh\hyperstack_ed25519 ubuntu@VM_IP
```
Step 2. When asked "Are you sure you want to continue connecting (yes/no/[fingerprint])?" type yes and press Enter.
You should see: a welcome text and the prompt ubuntu@medscriptai-gpu:~$
Step 3. Type and press Enter:
```
nvidia-smi
```
You should see: a table containing "NVIDIA A100 80GB PCIe", "Driver Version: 570.xx" and "CUDA Version: 12.8".
Step 4. Type and press Enter:
```
df -h / /ephemeral
```
You should see: the root disk / (about 100 GB) and a large disk mounted at /ephemeral [6].
Step 5. Type exit and press Enter (you are back on your laptop).
| Placeholder | Replace with | Example |
| VM_IP | the PUBLIC IP from Section 2A.9 | 38.80.123.45 |
Watch out: /ephemeral is fast and big but its data can be lost when the VM is hibernated or deleted [6]. We only put re-downloadable caches there. Your project lives in /home/ubuntu.
CHECKPOINT: nvidia-smi showed the A100 and driver 570 or newer.
If not: see Section 2A.14.

## SECTION 2A.11 - Connect VS Code to the GPU computer (Remote-SSH)
WHERE: VS Code on your laptop
GOAL: VS Code opens folders and terminals ON the GPU computer.
Step 1. Press F1 (a command box opens at the top).
Step 2. Type Remote-SSH: Open SSH Configuration File and press Enter.
Step 3. Click the first file: C:\Users\YOUR_NAME\.ssh\config
Step 4. Paste this at the end of the file (replace VM_IP and YOUR_NAME):
```
Host medscriptai
    HostName VM_IP
    User ubuntu
    IdentityFile C:\Users\YOUR_NAME\.ssh\hyperstack_ed25519
    ServerAliveInterval 60
```
Step 5. Press Ctrl+S to save.
Step 6. Press F1.
Step 7. Type Remote-SSH: Connect to Host and press Enter.
Step 8. Click medscriptai.
Step 9. If a box asks "Select the platform of the remote host", click Linux.
Step 10. Wait until the bottom-left corner shows SSH: medscriptai (the first time VS Code installs a small server on the VM, about 1 minute).
Step 11. In the top menu click Terminal > New Terminal.
You should see: a terminal panel at the bottom with the prompt ubuntu@medscriptai-gpu:~$
Step 12. In that terminal type hostname and press Enter.
You should see: medscriptai-gpu
| Line in the config | What it means |
| Host medscriptai | a short nickname you choose |
| HostName VM_IP | the real address |
| User ubuntu | the login name on Hyperstack Ubuntu images |
| IdentityFile ... | which private key to use |
| ServerAliveInterval 60 | send a small "I am here" message every 60 s so the connection does not drop |
Watch out: from now on, "VS Code Terminal" in this manual ALWAYS means this terminal connected to the GPU computer (bottom-left shows SSH: medscriptai), not your laptop.
CHECKPOINT: hostname prints medscriptai-gpu.
If not: F1 > Remote-SSH: Kill VS Code Server on Host > medscriptai, then repeat Steps 6-10.

## SECTION 2A.12 - Install and connect WinSCP (drag-and-drop files)
WHERE: web browser, then WinSCP on your laptop
GOAL: see the GPU computer's files next to your laptop files.
Step 1. Open https://winscp.net/eng/download.php
Step 2. Click DOWNLOAD WINSCP (the installation package).
Step 3. Run the downloaded WinSCP-x.x.x-Setup.exe, choose Install for me only, accept the licence, choose Typical installation, Commander interface, then Install and Finish.
Step 4. Open WinSCP. The Login window appears.
Step 5. Click New Site.
Step 6. Fill the fields:
| Field | Value | Why |
| File protocol | SFTP | file transfer inside SSH |
| Host name | VM_IP | your GPU computer |
| Port number | 22 | SSH port |
| User name | ubuntu | the login name |
| Password | (empty) | you use a key |
Step 7. Click Advanced...
Step 8. In the left list click SSH > Authentication.
Step 9. Next to Private key file click the ... button.
Step 10. In the file window change the file type filter (bottom right) to All Files (*.*).
Step 11. Open C:\Users\YOUR_NAME\.ssh\hyperstack_ed25519 (the file WITHOUT .pub).
Step 12. WinSCP asks to convert the key to PuTTY format: click OK, then Save (it creates hyperstack_ed25519.ppk).
Step 13. Click OK to close Advanced.
Step 14. Click Save, type medscriptai as the site name, click OK.
Step 15. Click Login.
Step 16. If asked "Continue connecting to an unknown server?", click Accept.
You should see: two panels: your laptop on the left, /home/ubuntu on the right.
How to copy files: drag a file from the right panel to the left panel = download to your laptop; drag from left to right = upload to the GPU computer.
CHECKPOINT: the right panel shows /home/ubuntu.
If not: "Server refused our key" = choose the .ppk file WinSCP created in Step 12.

## SECTION 2A.13 - Stop paying: hibernate, restore and delete
WHERE: Hyperstack console > Virtual Machines > click medscriptai-gpu
GOAL: no GPU bill while you are not working (do this EVERY time you stop).
Step 1. In the VS Code Terminal, check nothing important is running: type tmux ls (you will learn tmux in Part 2B; "no server running" is fine).
Step 2. In the Hyperstack console, open the VM details page (click medscriptai-gpu).
Step 3. Hover over the ⋮ menu in the top-right corner.
Step 4. Click Hibernate this VM.
Step 5. In the dialog, leave "Keep Public IP attached?" OFF (cheaper; you get a new IP later) or turn it ON (small hourly charge, same IP).
Step 6. Click Hibernate this VM to confirm.
You should see: status HIBERNATING, then HIBERNATED.
Step 7. To work again: open the VM details page, hover over ⋮ and click Restore (wait for ACTIVE).
Step 8. If the public IP changed: put the new IP in HostName of C:\Users\YOUR_NAME\.ssh\config (Section 2A.11 Step 1-5) and in the WinSCP site (Edit > Host name).
| VM status | GPU billed? | What is kept |
| ACTIVE | yes | everything |
| SHUTOFF (Stop) | YES, still billed [3] | everything |
| HIBERNATED | no (only storage and a kept IP are billed) [3] | root disk (/home/ubuntu); /ephemeral may be lost [6] |
| Deleted | no | nothing |
Watch out: "Stop" is NOT a money saver on Hyperstack. Use Hibernate this VM or Delete.
Watch out: restoring needs the GPU flavor to be in stock; if it is not, try later or deploy a new VM and copy your work from the Hugging Face Hub.
CHECKPOINT: when you finish a work session, the VM shows HIBERNATED.

## SECTION 2A.14 - Connection troubleshooting
WHERE: PowerShell, VS Code or WinSCP on your laptop, and the Hyperstack console
GOAL: fix any connection problem in a few minutes.
Step 1. Find your symptom in the table and do the fix in the right column.
| Symptom | Likely cause | Exact fix |
| ssh: connect to host ... port 22: Connection timed out | SSH access off, wrong IP, or VM not ACTIVE | 1) VM status must be ACTIVE. 2) Firewall tab > Enable SSH access ON. 3) Copy the IP again from the PUBLIC IP column |
| Permission denied (publickey) | wrong key file or wrong user | use -i with the file WITHOUT .pub and the user ubuntu; the key must be imported in the SAME environment as the VM |
| WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED | new VM on an old IP | in PowerShell run ssh-keygen -R VM_IP, then connect again |
| VS Code stuck on "Setting up SSH host" | half-installed VS Code server | F1 > Remote-SSH: Kill VS Code Server on Host, connect again |
| WinSCP "Server refused our key" | key not in PuTTY format | use the .ppk file from Section 2A.12 Step 12 |
| nvidia-smi: command not found | wrong OS image (no driver) | delete the VM and deploy again with an image whose name contains R570 CUDA 12.8 |
CHECKPOINT: after the fix, Section 2A.10 Step 1 logs you in again.
