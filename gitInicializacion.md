Una buena practica de programaicion es configurar el nombre de usuario y algunas credenciales en la configuracion de git.

1. En primer lugar se realiza la identificacion global del equipo donde se va a trabajar: Iniciar la app gitbash

	1.1 Consultar la onfiguracion por defecto de git
		>> git config --list
		>> git config --list --show-origin     # para ver donde estan guardadas las configuraciones
	
	1.2 Modificar el usuario global con el que se identificara el ordenador
		>> git config --global user.name "name"     # entre comillas

	1.3 Modificar el correo global con el que se identificara el ordenador
		>> git config --global user.email "email"     # entre comillas

	
2. Configuracion de git con github para enlazar repositorios

	2.1 Crear repositorios
		2.1.1 Unirse al repositorio remoto de de la organizacion en Github: https://github.com/UN-Bogota/GEE_python
		2.1.2 Crear un repositorio local en el ordenador donde se va a almacenar el proyecto.     # Carpeta
	
	2.2 Configuracion de las llaves publicas y privadas (protocolo SSH)
	
		2.2.1 Situarse en el home del ordenador
			>> cd 
		2.2.2 Congigurar el email en global
			>> git config --list     # Este paso es de verificacion, si el correo en "user.email" es el mismo 
							   que el configurado en el paso 1.3 no hay que modificarlo de nuevo 
			
			
		2.2.3 Generar las llaves, estando en la carpeta home
			>>ssh-keygen -t rsa -b 4096 -C "correo@gmail.com"

			2.2.3.1 elegir una carpeta, se recomienda que sea en el home (en windows solamente darle enter)
			2.2.3.2 crear una contraseña passphrase (contraseña adicional, se recomienda poner una)
		
		2.2.4 Agregar las llaves al entorno local: "id_rsa.pub" -> publica y "id_rsa" -> privada

			2.2.4.1 ir a la carpeta donde se guardo la llave (home si se siguieron las instrucciones del 
					paso 2.2.3.1 y abrir el archivo "id_rsa.pub" (abrir con un programa lector de 
					texto) y copiar todo el contenido. NOTA: En mac hay que hacer un paso adicional que 
					y es agregar la llave publica al entorno local para que el sistema la detecte (en Windows y Linux se hace 			 
					automatico)
			2.2.4.2 Revisar que el servidor de llaves este prendido
				>> eval $(ssh-agent -s)
			2.2.4.3 Agregar la llave al sistema
				>> ssh-add ~/.ssh/id_rsa     # esto agrega la llave privada

		2.2.5. Enlazar cuenta de github con el ordenador
		
			2.2.5.1 ir al perfil de github > settings > SSH and GPG keys > new ssh key
			2.2.5.2 se pone un titulo descriptivo del pc al que se va a conectar y se pone la llave ssh publica
		  		  almacenada en local que se copio en portapapeles
		    
		2.2.6 Enlazar la llave publica con el repositorio remoto
			
			2.2.6.1 ir al repositorio remoto en Github > boton "<>Code" > SSH > copiar enlace
			2.2.6.2 enlazar repositorio local con repositorio remoto: ir al gitbash > ir a la carpeta donde se creo el repositorio local
				  en el paso 2.1.2 con el comando >> cd "ruta del repositorio"     # sin comillas
				>> git remote add origin <<url>>>    # La url es la que se copio en la pagina de github
			2.2.6.3 git remote -v # verificar que la conexion se realizó.

Listop C:
		
		
