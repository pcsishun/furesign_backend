import sys
import json

dataToSendBack = '''{
                        "name": "210919_furesign_backend",
                        "version": "0.1.0",
                        "private": true,
                        "postinstall": "npm run build",
                        "start": "node server.js",
                        "scripts": {
                            "serve": "vue-cli-service serve",
                            "build": "vue-cli-service build"
                        },
                        "dependencies": {
                            "@splidejs/vue-splide": "^0.3.5",
                            "@types/cors": "^2.8.12",
                            "@types/express": "^4.17.13",
                            "@types/jquery": "^3.5.6",
                            "@types/multer": "^1.4.7",
                            "@types/mysql": "^2.15.19",
                            "@types/node": "^16.9.0",
                            "axios": "^0.21.4",
                            "body-parser": "^1.19.0",
                            "child_process": "^1.0.2",
                            "core-js": "^3.6.5",
                            "cors": "^2.8.5",
                            "dotenv": "^10.0.0",
                            "express": "^4.17.1",
                            "jquery": "^3.6.0",
                            "multer": "^1.4.3",
                            "mysql": "^2.18.1",
                            "serve-static": "^1.14.1",
                            "typescript": "^4.4.3"
                        },
                        "devDependencies": {
                            "node-sass": "^6.0.1",
                            "sass-loader": "^12.1.0"
                        }
                        }
                        '''
data = json.loads(dataToSendBack)
print(data)
sys.stdout.flush()

