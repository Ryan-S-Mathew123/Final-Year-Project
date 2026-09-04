# Bird Microphone Node

A Flutter mobile client that captures microphone audio and streams it to a bird-localization server over WebSockets. Each device identifies itself with a phone ID before sending audio data.

## Requirements

- Flutter SDK 3.x or newer
- Dart SDK 3.0 or newer
- Android Studio and an Android device/emulator, or Xcode and an iOS device/simulator
- A WebSocket server that accepts the audio stream

Check that Flutter is installed correctly:

```bash
flutter doctor
```

Resolve any required Android, iOS, or Flutter SDK issues reported by that command before running the app.

## Android Studio setup

Android Studio is required for Android SDK tools, an Android emulator, and Android device debugging. Install the latest stable version from [developer.android.com/studio](https://developer.android.com/studio).

During installation, make sure these components are selected:

- Android SDK
- Android SDK Platform for a recent stable Android version
- Android SDK Build-Tools
- Android SDK Platform-Tools, including `adb`
- Android Emulator
- Android SDK Command-line Tools

After installation:

1. Open Android Studio and go to **More Actions > SDK Manager**. Confirm the SDK Platform and SDK Tools listed above are installed.
2. Open **More Actions > Virtual Device Manager**, create a phone emulator, download a system image if needed, and start the emulator.
3. Accept Android SDK licenses from a terminal:

	```bash
	flutter doctor --android-licenses
	```

	Press `y` to accept each license.

4. Verify the Android toolchain:

	```bash
	flutter doctor
	```

For a physical Android phone, enable **Developer options** and **USB debugging**, connect it with a USB cable, and approve the computer's RSA debugging prompt. Check that it appears in:

```bash
flutter devices
```

If Flutter cannot find the Android SDK, configure its location with:

```bash
flutter config --android-sdk <path-to-android-sdk>
```

## Setup

1. Clone the repository and enter the project directory:

	```bash
	git clone https://github.com/Ryan-S-Mathew123/Final-Year-Project.git
	cd Final-Year-Project
	```

	If you are using the `bird_mobile` branch:

	```bash
	git switch bird_mobile
	```

2. Install Flutter dependencies:

	```bash
	flutter pub get
	```

3. Connect a phone or start an emulator, then confirm Flutter can see it:

	```bash
	flutter devices
	```

## Run the app

Run on the currently selected device:

```bash
flutter run
```

Run on a specific device by using the device ID shown by `flutter devices`:

```bash
flutter run -d <device-id>
```

For a browser or desktop development run, use one of the available device IDs, for example:

```bash
flutter run -d chrome
```

Microphone capture is intended for a real mobile device. Browser and desktop support depends on the platform implementation provided by the `record` package.

## Configure the WebSocket server

The app opens with these defaults:

- Server address: `ws://192.168.1.100:8000/ws`
- Phone ID: `phone_1`

Change the server address and phone ID in the app before pressing **Start Recording**. When testing against a server on your laptop:

- Put the phone and laptop on the same Wi-Fi network.
- Use the laptop's local network IP address, not `localhost` or `127.0.0.1`.
- Make sure the server listens on the network interface and that its firewall allows the WebSocket port.
- Use `wss://` when the server requires TLS.

The client sends the following message first:

```text
PHONE_ID:<phone-id>
```

It then sends mono PCM16 audio chunks at 44,100 Hz over the same WebSocket connection.

## Microphone permissions

Allow microphone access when the app asks for it. If permission was denied previously, enable it in the device's app settings and try again.

For iOS builds, ensure `NSMicrophoneUsageDescription` is present in `ios/Runner/Info.plist` with a user-facing explanation. For Android builds, ensure the app has the `RECORD_AUDIO` permission in the Android manifest if your target configuration does not add it automatically.

## Useful commands

```bash
# Check Dart and Flutter code
flutter analyze

# Run tests
flutter test

# List connected devices
flutter devices

# Build an Android APK
flutter build apk
```

## Project structure

```text
lib/main.dart       Main Flutter UI and audio/WebSocket client
android/            Android project files
ios/                iOS project files
test/               Flutter tests
pubspec.yaml        Dependencies and Flutter project configuration
```
