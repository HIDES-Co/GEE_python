# GIT INITIALIZATION SETTINGS

## 1.  Global Git Identification and Download

1.1. Download and install git software from https://git-scm.com/downloads

1.2. Global identification of your pc
1.2.1. start git bash
1.2.2. consult your default configuration:
`$ git config --list`
if you want to show where are the git settings storage:
`$ git config --list --show-origin `
1.2.3. Set the global user on your pc
`$ git config --global user.name "git username"`

1.2.4. Set the global email on your pc
`$ git config --global user.email "email@..."`

## 2. Connect Git and Github

2.1 Set your public and private keys (SSH protocol)

2.1.1. Open git bash and locate on your home.
`$ cd`
2.1.2. Verify that your github email is registered on your git global settings.
`$ git config --list`
2.1.3. Now you will going to generate your keys being located in the home directory.
`$ ssh-keygen -t rsa -b 4096 -C "email@gmail.com"`
2.1.3.1. Choose where the keys are to be stored.
2.1.3.2. Create a passphrase (aditional pass, recomended).

2.1.4. Add the public and private keys:  
"id_rsa.pub" --> public and "id_rsa" --> private

2.1.4.1. Go to the folder where the keys are stored.
2.1.4.2. Open the "id_rsa.pub" file using a text editor and copy all it's content.

2.1.5 Verify that the keys server is open.
`$ eval $(ssh-agent -s)`
you should get:
`>>  Agent pid ###number###`

2.1.6. Add your keys in your system.
`$ ssh-add ~/.ssh/id_rsa`

2.3. Link the GitHub account to your computer.

2.3.1. Go to your profile settings in the HitHub page >> SSH and GPG keys >> new ssh key. (https://github.com/settings/keys).

2.3.2. Write a descriptive title about the computer that you will link and paste your public key (2.1.4.2.).


## 3. Clone the repository 

3.1. Go to the remote GitHub repo (https://github.com/UN-Bogota/GEE_python) >> button "<>Code" >> SSH >> copy.

3.2. Make a directory where you going to store the project and open git bash here. 
Ej:
write `$ pwd` in your terminal and you should get : `/home/User/Documents/New_Directory`

3.3 Clone the project whit ssh link (3.1).
`$ git clone <<ssh_url>>`

3.4 Finally, check that the conection was succeful.
`$ git remote -v`
You shoyld get:
`>> origin	git@github.com:UN-Bogota/GEE_python.git (fetch)`
`>> origin	git@github.com:UN-Bogota/GEE_python.git (push)`

That's all. :)

<img src="https://pngimg.com/uploads/github/github_PNG15.png" alt="drawing" width="200"/> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Escudo_de_la_Universidad_Nacional_de_Colombia_%282016%29.svg/598px-Escudo_de_la_Universidad_Nacional_de_Colombia_%282016%29.svg.png" alt="drawing" width="80"/>
