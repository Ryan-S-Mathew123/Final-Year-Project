import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  runApp(const BirdMicApp());
}

class BirdMicApp extends StatelessWidget {
  const BirdMicApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Bird Microphone',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const MicrophonePage(),
    );
  }
}

class MicrophonePage extends StatefulWidget {
  const MicrophonePage({super.key});

  @override
  State<MicrophonePage> createState() => _MicrophonePageState();
}

class _MicrophonePageState extends State<MicrophonePage> {
  final AudioRecorder _recorder = AudioRecorder();
  WebSocketChannel? _channel;
  StreamSubscription<Uint8List>? _audioSubscription;

  final TextEditingController _serverController =
      TextEditingController(text: 'ws://192.168.1.100:8000/ws');

  final TextEditingController _phoneIdController =
      TextEditingController(text: 'phone_1');

  bool _recording = false;
  String _status = 'Stopped';

  Future<void> startRecording() async {
    final permission = await Permission.microphone.request();

    if (!permission.isGranted) {
      setState(() => _status = 'Microphone permission denied');
      return;
    }

    try {
      _channel = WebSocketChannel.connect(
        Uri.parse(_serverController.text.trim()),
      );

      _channel!.sink.add('PHONE_ID:${_phoneIdController.text.trim()}');

      final stream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 44100,
          numChannels: 1,
          bitRate: 705600,
        ),
      );

      _audioSubscription = stream.listen(
        (Uint8List audioChunk) {
          _channel?.sink.add(audioChunk);
        },
        onError: (Object error) {
          if (mounted) {
            setState(() => _status = 'Audio error: $error');
          }
        },
      );

      setState(() {
        _recording = true;
        _status = 'Streaming audio continuously...';
      });
    } catch (e) {
      setState(() => _status = 'Connection error: $e');
    }
  }

  Future<void> stopRecording() async {
    await _audioSubscription?.cancel();
    await _recorder.stop();
    await _channel?.sink.close();
    _channel = null;

    if (mounted) {
      setState(() {
        _recording = false;
        _status = 'Stopped';
      });
    }
  }

  @override
  void dispose() {
    _audioSubscription?.cancel();
    _recorder.dispose();
    _channel?.sink.close();
    _serverController.dispose();
    _phoneIdController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Bird Microphone Node')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('Phone ID',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            TextField(controller: _phoneIdController),
            const SizedBox(height: 20),
            const Text('Laptop WebSocket Address',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            TextField(
              controller: _serverController,
              keyboardType: TextInputType.url,
            ),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: _recording ? stopRecording : startRecording,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.all(20),
              ),
              child: Text(_recording ? 'STOP RECORDING' : 'START RECORDING'),
            ),
            const SizedBox(height: 30),
            Center(
              child: Text(
                _status,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 18,
                  color: _recording ? Colors.green : Colors.red,
                ),
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              'Keep this app open during the demonstration. '
              'All phones and the laptop must be connected to the same Wi-Fi network.',
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
