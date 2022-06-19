from flask import Flask, request, jsonify, make_response
from flask_restful import marshal_with, Resource, reqparse, fields
from flask_restful import Api
import requests
import os
 
application = Flask(__name__)
api = Api(application)
 
botToken = os.environ['TOKEN'] ##반드시 본인 챗봇의 토큰을 입력하시오
 
class Telebot(Resource):
    s = requests.session()
    sendUrl = "https://api.telegram.org/bot{}".format(botToken)
 
    #https://example.com/telebot 경로로 들어온 트레픽중 post메서드를 수신하는 부분
    def post(self):
        getText = request.json['message']['text'].strip() #사용자가 입력한 체팅
        chatId = request.json['message']['chat']['id'] #체팅방 id
        self.sendMessage(chatId, getText)
 
    #sendMessage API로 체팅방에 메시지 보내는 함수
    def sendMessage(self, chatId, getText):
            sendUrl = "{}/sendMessage".format(self.sendUrl)
            params = {
                "chat_id" : chatId,
                "text" : getText
            }
            self.s.get(sendUrl, params=params)
            return None
 
#https://example.com/telebot 경로로 들어오면 Telebot 클래스로 전달
api.add_resource(Telebot, '/telebot')
 
if __name__ == '__main__':    
    context = ('./certs/fullchain.pem', './certs/privkey.pem') ##반드시 본인 도메인의 인증서를...
    application.run(debug = True, host='0.0.0.0', port=8443, threaded=True, ssl_context=context)