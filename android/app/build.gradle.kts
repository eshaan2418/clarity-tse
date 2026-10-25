plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.clarity.app"
    compileSdk = 35
    ndkVersion = "27.2.12479018"
    defaultConfig {
        applicationId = "org.clarity.app"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1-research"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild { cmake { cppFlags += listOf("-std=c++17", "-O2"); arguments += listOf("-DANDROID_STL=c++_shared", "-DORT_DIR=" + layout.buildDirectory.dir("ort").get().asFile.absolutePath) } }
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }
    buildFeatures { prefab = true }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging { jniLibs { useLegacyPackaging = false } }
    androidResources { noCompress += listOf("onnx") }
}

// onnxruntime-android has no prefab: unpack its headers + .so for the native (JNI) engine
val ortAar: Configuration by configurations.creating

dependencies {
    ortAar("com.microsoft.onnxruntime:onnxruntime-android:1.20.0@aar")
    implementation("com.google.oboe:oboe:1.9.3")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
}

val extractOrt by tasks.registering(Copy::class) {
    from({ zipTree(ortAar.singleFile) }) { include("headers/**", "jni/**") }
    into(layout.buildDirectory.dir("ort"))
}
tasks.matching { it.name.startsWith("configureCMake") || it.name.startsWith("buildCMake") }
    .configureEach { dependsOn(extractOrt) }
